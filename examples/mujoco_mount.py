"""Small machined mounting plate with chamfers, through holes and fasteners."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def plate_mesh(half=.12, height=.012, chamfer=.014, hole_radius=.011, through_radius=.0045):
    outline = np.array([[-half, -half+chamfer], [-half+chamfer, -half],
                        [half-chamfer, -half], [half, -half+chamfer],
                        [half, half-chamfer], [half-chamfer, half],
                        [-half+chamfer, half], [-half, half-chamfer]])
    centers = np.array([[x, y] for x in (-.10, .10) for y in (-.10, .10)])
    angles = np.arange(24)*2*np.pi/24
    def holes_at(radius):
        return np.concatenate([center+radius*np.column_stack([np.cos(angles), np.sin(angles)])
                               for center in centers])

    holes, through = holes_at(hole_radius), holes_at(through_radius)
    rings = [(outline*.992, 0.), (outline, .001),
             (outline, height-.001), (outline*.992, height)]
    vertices = np.concatenate([np.column_stack([xy, np.full(8, z)]) for xy, z in rings])
    top_holes = np.column_stack([holes, np.full(len(holes), height)])
    # The washer rests on a counterbore shoulder, with an M8 clearance
    # hole through the remaining plate thickness.
    shoulder = height-.007
    bore_shoulder = np.column_stack([holes, np.full(len(holes), shoulder)])
    through_shoulder = np.column_stack([through, np.full(len(holes), shoulder)])
    bottom_holes = np.column_stack([through, np.zeros(len(holes))])
    vertices = np.concatenate([vertices, top_holes, bore_shoulder, through_shoulder, bottom_holes])
    faces = []
    for ring in range(3):
        for i in range(8):
            a, b = ring*8+i, ring*8+(i+1) % 8
            faces.extend([[a, b, b+8], [a, b+8, a+8]])
    for offset, outer, reverse, radius in ((32, np.arange(24, 32), False, hole_radius),
                                           (32+3*len(holes), np.arange(8), True, through_radius)):
        ids = np.r_[outer, np.arange(offset, offset+len(holes))]
        points = vertices[ids, :2]
        triangles = Delaunay(points).simplices
        for triangle in triangles:
            center = points[triangle].mean(axis=0)
            if np.any(np.linalg.norm(centers-center, axis=1) < radius*.999):
                continue
            face = ids[triangle].tolist()
            faces.append(face[::-1] if reverse else face)
    for bottom_ring, top_ring in ((3, 2), (2, 1), (1, 0)):
        for hole in range(4):
            for i in range(24):
                a = 32+bottom_ring*len(holes)+hole*24+i
                b = 32+bottom_ring*len(holes)+hole*24+(i+1) % 24
                c = 32+top_ring*len(holes)+hole*24+i
                d = 32+top_ring*len(holes)+hole*24+(i+1) % 24
                faces.extend([[a, c, d], [a, d, b]])
    return vertices, np.asarray(faces), centers


def pedestal_mesh(asset, height, offset):
    """Follow the actual mounting-foot outline, with a 2 mm support margin."""
    vertices = []
    for mesh in asset.findall("mesh"):
        if mesh.get("name", "").startswith("full_fr3v2_link0_"):
            for line in Path(mesh.attrib["file"]).read_text().splitlines():
                if line.startswith("v "):
                    vertices.append([float(value) for value in line.split()[1:4]])
    vertices = np.asarray(vertices)
    foot = vertices[vertices[:, 2] < vertices[:, 2].min()+.001, :2]-offset[:2]
    outline = foot[ConvexHull(foot).vertices]
    angles = np.arange(32)*2*np.pi/32
    margin = .002*np.column_stack([np.cos(angles), np.sin(angles)])
    expanded = (outline[:, None, :]+margin[None, :, :]).reshape(-1, 2)
    outline = expanded[ConvexHull(expanded).vertices]
    count = len(outline)
    vertices = np.vstack([np.column_stack([outline, np.zeros(count)]),
                          np.column_stack([outline, np.full(count, height)])])
    faces = []
    for i in range(1, count-1):
        faces.extend([[0, i+1, i], [count, count+i, count+i+1]])
    for i in range(count):
        j = (i+1) % count
        faces.extend([[i, j, j+count], [i, j+count, i+count]])
    return vertices, np.asarray(faces)


def add_mount(world, asset, table_z, base_z):
    # The FR3v2 mounting-foot bounds are centered behind its joint-frame origin.
    # Align the complete mounting assembly with the foot, not the arm axis.
    offset = np.array([-.04103731, 0., 0.])

    def numbers(values):
        return " ".join(f"{v:.8g}" for v in np.asarray(values).reshape(-1))

    def geom(name, kind, size, pos, material, **extra):
        return ET.SubElement(world, "geom", name=name, type=kind, size=numbers(size),
                             pos=numbers(np.asarray(pos)+offset), material=material, contype="0", conaffinity="0", **extra)

    ET.SubElement(asset, "material", name="mount_aluminum", rgba=".56 .61 .65 1",
                  specular=".75", shininess=".6")
    ET.SubElement(asset, "material", name="mount_dark", rgba=".09 .12 .15 1",
                  specular=".45", shininess=".4")
    ET.SubElement(asset, "material", name="mount_socket", rgba=".014 .021 .028 1")
    vertices, faces, centers = plate_mesh()
    ET.SubElement(asset, "mesh", name="mount_plate_mesh", vertex=numbers(vertices),
                  face=" ".join(map(str, faces.reshape(-1))))
    ET.SubElement(world, "geom", name="mount_plate", type="mesh", mesh="mount_plate_mesh",
                  pos=numbers(np.array([0, 0, table_z])+offset), material="mount_aluminum", contype="0", conaffinity="0")
    pedestal_height = base_z-table_z-.012
    vertices, faces = pedestal_mesh(asset, pedestal_height, offset)
    ET.SubElement(asset, "mesh", name="mount_pedestal_mesh", vertex=numbers(vertices),
                  face=" ".join(map(str, faces.reshape(-1))))
    ET.SubElement(world, "geom", name="mount_pedestal", type="mesh", mesh="mount_pedestal_mesh",
                  pos=numbers(np.array([0, 0, table_z+.012])+offset), material="mount_dark",
                  contype="0", conaffinity="0")
    geom("mount_registration_ring", "cylinder", [.079, .001], [0, 0, base_z-.001], "mount_aluminum")
    for i, (x, y) in enumerate(centers):
        geom(f"mount_bolt_{i}_shaft", "cylinder", [.004, .012], [x, y, table_z], "mount_dark")
        geom(f"mount_bolt_{i}_washer", "cylinder", [.010, .0007], [x, y, table_z+.0057], "mount_aluminum")
        geom(f"mount_bolt_{i}_head", "cylinder", [.007, .0025], [x, y, table_z+.0089], "mount_dark")
        geom(f"mount_bolt_{i}_socket", "cylinder", [.0035, .0002], [x, y, table_z+.0116], "mount_socket")
    for y in (-.045, .045):
        geom(f"mount_dowel_{y}", "cylinder", [.003, .0015], [-.10, y, table_z+.012], "mount_aluminum")
