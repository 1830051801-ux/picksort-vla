# Real Robot Migration

The simulator demonstrates a vision-to-coordinate-to-grasp/sort control flow, but its camera geometry, arm hardware and grasp adapter are simulation-specific. Migration should preserve the verified contracts and replace those adapters one boundary at a time.

## Preserve these contracts

- Publish one stable `sorting_cell_interfaces/msg/DetectedObject` per workpiece in `base_link`.
- Accept complete trajectories through `control_msgs/action/FollowJointTrajectory`.
- Publish fresh, correctly named `sensor_msgs/msg/JointState` feedback.
- Report positive or negative gripper acknowledgement before transfer or retreat continues.
- Keep `CellStatus`, routing rules, workspace validation and terminal CSV metrics.

Preserving these semantics lets the planner and coordinator remain unchanged while each simulator dependency is replaced.

## Replace the simulator adapters

| Simulation component | Real-cell replacement |
| --- | --- |
| Gazebo RGB camera and fixed workplane projection | Calibrated industrial/USB camera, validated intrinsics, distortion correction and depth or measured workplane intersection |
| Simulator `base_link` projection | Hand-eye calibration plus timestamped TF2 camera-to-base transformation |
| HSV scene detector | Reuse only if lighting/parts are controlled; otherwise use the qualified detector while preserving `DetectedObject` |
| `gz_ros2_control` system | Robot manufacturer's `ros2_control` hardware interface or a trajectory bridge exposing `FollowJointTrajectory` |
| Gazebo `/joint_states` | Encoder-derived robot and gripper feedback from the real driver |
| `gazebo_attachment` and `SetEntityPose` | Real gripper action/I/O with close, object-present and release acknowledgement |
| Gazebo bins and workspace | Surveyed cell coordinates and guarded physical fixtures |

The current default perception is already image-based; migration is not simply “replace synthetic perception.” The necessary change is replacing the simulator camera/projection assumptions and qualifying the detector under real optics and lighting.

## Recommended migration sequence

1. **Inventory the hardware interfaces.** Confirm joint names, units, action server name, feedback rate, gripper protocol, emergency-stop chain and safe torque off.
2. **Bring up measured state only.** Verify joint sign, zero offset, limits and timestamps without enabling motion.
3. **Qualify the trajectory action.** Send low-speed single-joint and home goals, checking acceptance, result codes and measured final error.
4. **Calibrate the tool.** Measure TCP, payload and finger geometry, then update kinematics and grasp offsets.
5. **Calibrate vision.** Validate intrinsics and camera-to-base transform against measured points across the entire pickup region.
6. **Integrate gripper acknowledgement.** Do not report success on command transmission alone; require controller completion and, where available, object-present/current/position evidence.
7. **Run dry cycles.** Publish detections and generate plans while motion output is inhibited; compare every target with surveyed coordinates.
8. **Enable reduced-speed motion.** Start with one fixed target, no part, then a single part and one destination.
9. **Expand the matrix.** Exercise all pickup extremes, three routes, recovery paths and repeated cycles before normal speed.

## Before enabling hardware motion

- install and test the hardware emergency stop, protective stop and controller watchdog;
- enforce manufacturer joint, velocity, acceleration, payload and tool limits below the application layer;
- verify the commanded `base_link`, tool frame, joint order and units with an independent measurement;
- start at no more than 10 percent of qualified velocity and acceleration;
- clear the workspace and use a guarded manual enable during first motion;
- reject stale transforms, stale detections and stale joint feedback;
- test network loss, action rejection, action timeout, feedback mismatch and gripper failure;
- confirm that every failure prevents subsequent transfer motion and emits one terminal `FAILED` status.

## Gripper acknowledgement boundary

The simulation pauses after `GRASP` until it receives `attached:<id>:...`, and after `RELEASE` until `detached:<id>:...`. A production driver does not have to use the text topic protocol, but it must preserve the behavior:

- **grasp success:** commanded close completed and the defined grasp-success sensor or threshold is true;
- **grasp failure:** timeout, controller fault, no-object condition or contradictory feedback;
- **release success:** commanded open completed and object-present feedback is clear;
- **release failure:** timeout or contradictory feedback.

If a gripper exposes a ROS action, adapt its result to this semantic boundary instead of acknowledging immediately when the goal is sent.

## Hardware acceptance evidence

Create a hardware verifier separate from the Gazebo named-pose check. At minimum, record:

- camera and joint-feedback rates plus timestamp/latency bounds;
- raw detection, transformed pose and calibration version;
- trajectory goal/result and measured final joint error for every segment;
- gripper command, acknowledgement and object-present evidence;
- final route confirmation from a bin sensor or independently measured observation;
- one unique terminal event and metrics row per object;
- emergency-stop, timeout and rejected-goal recovery results.

The committed simulation baseline is [`evidence/ros2_runtime_verification.json`](evidence/ros2_runtime_verification.json). It is useful for regression comparison, but it is not evidence that a physical cell is safe or production-ready.

## Safety boundary

The application-level checks are defense in depth, not a safety-rated control system. Hardware interlocks, risk assessment, guarding, safe motion limits and the manufacturer's controller protections remain authoritative.
