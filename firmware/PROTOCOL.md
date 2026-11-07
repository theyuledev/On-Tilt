# serial protocol

usb serial at 115200 baud. one command per line, one reply line per command. pulses are in us.

| send | reply | what |
|---|---|---|
| `S` | `OK S <armed 0/1> <limits 0/1> <ms since last cmd> <fw version>` | status |
| `L xlo xhi ylo yhi xlevel ylevel` | `OK L`, `ERR L envelope`, `ERR L armed` | limits + level pulse per servo. every value inside the board's envelope (1050-1950), lo < hi, level between lo and hi. not while armed |
| `A` | `OK A`, `ERR A nolimits`, `ERR A ramping` | arm, pulses start at level. needs an accepted `L` first |
| `D` | `OK D` | disarm, pulses stop |
| `P x y` | `OK P x y`, `ERR P disarmed` | set pulses, reply has the clamped values |
| `M 0/1` | `OK M` | marker led off/on |
| anything else | `ERR ?` | |

- boots disarmed, no pulses
- any command counts as the laptop being alive
- nothing for 250 ms while armed: ramps both servos to level at 300 us/s, holds level 40 ms so it actually goes out, then stops pulses. stays disarmed until the next `A`
- watchdog resets the board if the loop hangs for 500 ms, it comes back disarmed
- pins: x servo d9, y servo d10, marker led d7
