# decisions

running notes on the calls we made and why. newest at the bottom. add to it when something gets decided so nobody has to re-argue it at 3 am.

## sat nov 7

**~3 pm, contracts first.** before anyone writes real code we agreed the shared vocabulary. plate frame origin is the centre of the plate, +x right and +y up in the overhead image. servo pulses are in microseconds, commands go out as tilt degrees. every number lives in settings with a label saying where it came from: measured, sim-tuned, or untested default. no magic constants in code.

**~4:30 pm, kill switch before any servo moves.** nothing gets powered until the kill layers are checked. key kill eases the plate back to level then disarms, and the board self levels on its own if it is armed and goes 250 ms without a command. both verified before the first servo test.

**5:24 pm, camera settings locked.** 1280x800 at 120 fps with 3 ms manual exposure, auto exposure off so the ball doesnt smear. lens calibrated, 0.29 px rms reprojection, good enough.

**6:07 pm / 7:57 pm, servo bench.** benched servos 1 to 3. dead band is 12 to 20 us, delay is 44 to 48 ms including the camera, slew is about 110 deg/s. mounting 1 and 2, not bothering to bench 4 to 6, out of time. only buying better servos if balancing actually fails with these.

**7:41 pm, merging while the frame gets built.** safety prs on the app side (41, 57, 65, 76) merge on the reviewers pass while the hardware lead is heads down on the frame, he reads them after. anything that touches firmware still waits for him.

**7:44 pm, camera remounted, real clips.** camera is now centred over the plate with the whole plate in view. recorded three hand rolled clips (squash ball, lacrosse ball, small ball) and tuned detection against those instead of guessing thresholds.

**8:10 pm, servo tool finds the horn by motion.** rebuilt the servo tool so it finds the horn by what moves, not by shape. there is a fake servo test that proves the numbers come out right. no more shape filters on sticky notes.

**8:15 pm, plate pose from aruco.** four aruco markers on the plate corners plus two on the table so a table shake doesnt read as a tilt. ball position comes from intersecting the camera ray with the plane at ball height. markers printed at 50 mm.

**8:20 pm, sim gets the measured servo numbers.** dead band, delay and slew from the bench go into the sim. the pd has to show 45 deg phase margin with the measured delay before it is allowed to run on the real plate.

**8:30 pm, research then build, software dead band.** anything big gets researched, written up and checked by someone else before it gets built. dead band gets handled in software with an inverter plus a small integral term rather than swapping servos.
