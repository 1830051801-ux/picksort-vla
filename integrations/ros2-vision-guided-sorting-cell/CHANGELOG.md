# Changelog

## 1.0.0 — 2026-07-17

- Added camera-based blue, red and orange part detection in the Gazebo scene.
- Switched motion execution to `FollowJointTrajectory` goals with measured
  joint-state verification for each workflow segment.
- Added acknowledged Gazebo attachment handling and final named-model bin checks.
- Added fail-stop action cancellation, one-command Windows/Linux runners and a
  headless ROS 2 end-to-end verifier.
- Added ROS/Gazebo CI, sanitized reference evidence and migration documentation.
