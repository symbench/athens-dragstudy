Is it possible to run the drag model for a single component of the vehicle  (e.g. prop, cylinder, wing)?

- single prop + motor: works (error line 1494 concatenate meshes out of bounds index without including motor); is  
- single tube: works
- single wing: works

- the flow_face() function on line 86 takes as input mesh facets and mesh normals to obtain the length, width, and thickness of the component mesh
- flow_face is called for the components: ESC, Flange, Plate, Wings, non AeroShell
- flow_face is NOT called for components: tube, motor, fuse_capsule, props

From Swri biweekly 08/22/22:
- "careful changing lengths directly in json file/ dictionary since Creo is not updating positions, these will not flow through to the attached components. For example, change length of an arm containing a motor, and the motors position as before is the same which will affect drag"
    - this will make it more difficult to get drag sweeps since we do not know exactly how to update translation/rotation for an arbitrary parameter change in a vehicle
- re: why not just closed form formulas? Why use meshes? "if something is not used now, it is possible that it will be relevant later when dealing with cargo and structural analysis pipelines" 


### Drag Overview Summary (note: work in progress)

⚠️ = a "mesh" (triangulated surface representing 3d geometry) is required for this step of the drag calculation

Drag is computed for a component in the following steps (some things may not be completely accurate as these notes are updated):

1. For every CAD part in the `designData.json` input file, a mesh2cad rotation matrix is constructed based on the component's CAD parameters.
2. From here, `trimesh.creation.shape()` is called using this matrix and parameters where shape is cylinder, capsule, box, etc. to create a mesh representing this CAD part (e.g. for a cylinder on line 592)
3. The variable `Tform` is an hstack of the rotation and translation matrices applied to the newly created mesh to position it the way it is oriented in the actual vehicle (e.g. on line 600)
4. If the component is a box shape, the front face of this mesh after orientation is then determined using mesh facet normals ⚠️: `front_index = np.argmax(mesh.facets_normal[:, 0])` and `flow_face(mesh.facets_origin, mesh.facets_normal)` is called to get the flow face dimensions (the size of the area that is moving through air during flight). If the component is not a box shape, the drag surrogate function is called directly (e.g. `cylinderdrag()`).
5. If applicable, the flow face dimensions are then provided as input to the drag surrogate function, for example `cd, cl, cf, warea = platedrag( leng / 1000, thi / 1000, wid / 1000, mesh.facets_normal[front_index,], ang, vel, mu, rho, q)`. Again `flow_face()` is only called for "box-like" objects (ESC, Flange, Plate, Wing). It is not called for cylinder or ellipsoid drag surrogate functions which take as input CAD parameters, the mesh axis of symmetry ⚠️, and angle, velocity, and airflow constants. A drag surrogate may sometimes but not always require a mesh.
6. The drag surrogate function provides `cd, cl, cf, warea`, which are drag values (`cd` seems most important) and reference area for the current CAD part and its representative mesh.
7. Next, a projection of this mesh ⚠️ onto the YZ plane is added to a list of polygons (line 1426) which are later used to check for pairwise intersections to determine how these interactions account for overall drag (around line 1490)
8. The loop on line 1484 iterates over pairwise polygons (projections onto YZ plane from meshes), checks if they intersect, and then uses characteristic length and euclidean distance based on bounding box and centers of gravity to determine a ratio (?) about the fraction of intersected area - it looks like this loop obtains a 'modification scalar' that indicates the scale of the overall drag that contributes from the overlapping area of a pair of polygons
9. The actual drag is calculated as a cumulative sum within the final loop on line 1571, where each drag term is a Navier-Stokes expression (?) based on air constants (e.g. `rho`) as well as the `cd` drag value from above for each CAD part surrogate. There is also an additional drag term for what looks like a velocity of 5m/s (`vp = 5` on line 486) and an angle of attack of 4deg (`ap = 4` on line 487). A single drag matrix has shape (10, 9) and `vp, ap` appear to index into the drag matrix at a cruising velocity/angle.
10. `Drag = TotalDrag - WingDrag` for each direction and finally `Drag_i = (2 * DragX / (rho * Vx**2)) * 1000**2`
