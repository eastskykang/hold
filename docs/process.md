# Data processing

In this document, the term "bone" is synonymous with a "joint" in the `.bvh` file format specification.

## Estimated foot bones

The original foot bones output by the motion capture software do not provide clear information about the feet's interaction with the ground plane, and they may be subject to errors caused by miscalibration or marker movement.
For this reason, extra "Estimated" bones have been added to the files:

- `LeftSoleHeel_Estimated`
- `RightSoleHeel_Estimated`
- `LeftSoleToe_Estimated`
- `RightSoleToe_Estimated`

The positions and orientations of these bones are configured such that, when the actor is standing with flat feet in double support:

- The bones lie flat on the ground, at a height of approximately $z=0$.
- The Z axis of the bone frame is aligned with gravity.
- The X axis of the bone frame points in the direction of the toes.

To obtain these poses, we detect double support using a velocity and height thresholding heuristic on the foot and toe bones, then estimate the constant foot bone pose that minimizes several cost terms across all detected double-support states.
This process is not always perfect: the motion capture volume may not have a correct floor calibration, and the heuristic itself can fail.
For files where the heuristic failed, we manually specified that the estimate from another skeleton in a similar take be reused.

## Prop bones

Some files include motion-captured props. These are boxes and, in some files, chairs. The extents of the props were not recorded precisely and have been inferred.
The prop bones are parented to the `Z_UP_WORLD_ORIGIN` bone. Each prop bone has child bones that represent the 8 corners of its bounding box via their end site offsets.
If you import these `.bvh` files into Blender, the extents of the boxes are clearly indicated by the tips of the bones.
To deduce the size of a box directly from the `.bvh` file, read the end site offset of the `<prop_name>_x_y_z_corner` bone, which contains the half-lengths of the box. These are expressed in the frame of the prop bone (named `<prop_name>`).

## Temporal adjustment for reduced-scale BVH

In the scaled data, all lengths are at $80\%$ scale, and time runs at $1/\sqrt{0.8}$ the rate in order to preserve $g = 9.81\,\mathrm{m/s^2}$.
To avoid needless quality loss, the pose data was not resampled. Instead, the frame time in the scaled `.bvh` files was modified. Take this into account when reading these files, and interpolate poses appropriately if you wish to keep gravity at $9.81\,\mathrm{m/s^2}$ in the scaled data.
