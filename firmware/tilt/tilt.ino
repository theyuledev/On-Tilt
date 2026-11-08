#include <Servo.h>
#include <WDT.h>

const char FW_VERSION[] = "0.2";

const int SERVO_X_PIN = 9;
const int SERVO_Y_PIN = 10;
const int MARKER_LED_PIN = 7;

const int ENVELOPE_LO_US = 1050;
const int ENVELOPE_HI_US = 1950;
const unsigned long TIMEOUT_MS = 250;
const float RAMP_US_PER_MS = 0.3;
const unsigned long HOLD_LEVEL_MS = 40;
const unsigned long WATCHDOG_MS = 500;

Servo servoX, servoY;

bool armed = false;
bool limitsLoaded = false;
bool ramping = false;
int xLo, xHi, yLo, yHi, xLevel, yLevel;
float x, y;
unsigned long lastCmdMs = 0;
unsigned long sinceLastCmdMs = 0;
unsigned long lastLoopMs = 0;
unsigned long levelSinceMs = 0;

char line[64];
size_t lineLen = 0;
bool lineTooLong = false;

void stopPulses() {
  servoX.detach();
  servoY.detach();
  armed = false;
  ramping = false;
}

void writePulses() {
  servoX.writeMicroseconds((int)x);
  servoY.writeMicroseconds((int)y);
}

int clampInt(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

float toward(float cur, float target, float step) {
  if (fabs(target - cur) <= step) return target;
  return target > cur ? cur + step : cur - step;
}

bool inEnvelope(int v) {
  return v >= ENVELOPE_LO_US && v <= ENVELOPE_HI_US;
}

bool validAxis(int lo, int hi, int level) {
  return inEnvelope(lo) && inEnvelope(hi) && lo < hi && level >= lo && level <= hi;
}

void handle(char *cmd) {
  char *name = strtok(cmd, " ");
  int v[6];
  int n = 0;
  for (char *t = strtok(NULL, " "); t && n < 6; t = strtok(NULL, " ")) v[n++] = atoi(t);

  if (!name || strlen(name) != 1) {
    Serial.println("ERR ?");
    return;
  }

  char buf[48];
  switch (name[0]) {
    case 'S':
      snprintf(buf, sizeof buf, "OK S %d %d %lu %s %d", armed, limitsLoaded, sinceLastCmdMs, FW_VERSION, ramping);
      Serial.println(buf);
      break;

    case 'L':
      if (n != 6) { Serial.println("ERR ?"); break; }
      if (armed) { Serial.println("ERR L armed"); break; }
      if (!validAxis(v[0], v[1], v[4]) || !validAxis(v[2], v[3], v[5])) {
        Serial.println("ERR L envelope");
        break;
      }
      xLo = v[0]; xHi = v[1]; yLo = v[2]; yHi = v[3]; xLevel = v[4]; yLevel = v[5];
      limitsLoaded = true;
      Serial.println("OK L");
      break;

    case 'A':
      if (!limitsLoaded) { Serial.println("ERR A nolimits"); break; }
      if (ramping) { Serial.println("ERR A ramping"); break; }
      if (!armed) {
        x = xLevel;
        y = yLevel;
        servoX.attach(SERVO_X_PIN);
        servoY.attach(SERVO_Y_PIN);
        writePulses();
        armed = true;
        ramping = false;
      }
      Serial.println("OK A");
      break;

    case 'D':
      stopPulses();
      Serial.println("OK D");
      break;

    case 'P':
      if (n != 2) { Serial.println("ERR ?"); break; }
      if (!armed || ramping) { Serial.println("ERR P disarmed"); break; }
      x = clampInt(v[0], xLo, xHi);
      y = clampInt(v[1], yLo, yHi);
      writePulses();
      snprintf(buf, sizeof buf, "OK P %d %d", (int)x, (int)y);
      Serial.println(buf);
      break;

    case 'M':
      if (n != 1) { Serial.println("ERR ?"); break; }
      digitalWrite(MARKER_LED_PIN, v[0] ? HIGH : LOW);
      Serial.println("OK M");
      break;

    default:
      Serial.println("ERR ?");
  }
}

void readSerial(unsigned long now) {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c != '\n') {
      if (lineLen < sizeof line - 1) line[lineLen++] = c;
      else lineTooLong = true;
      continue;
    }
    line[lineLen] = '\0';
    sinceLastCmdMs = now - lastCmdMs;
    lastCmdMs = now;
    if (lineTooLong) Serial.println("ERR ?");
    else handle(line);
    lineLen = 0;
    lineTooLong = false;
  }
}

void rampToLevel(unsigned long now) {
  float step = RAMP_US_PER_MS * (now - lastLoopMs);
  x = toward(x, xLevel, step);
  y = toward(y, yLevel, step);
  writePulses();

  // servos hold wherever the last pulse put them, so make sure level actually goes out
  if (x == xLevel && y == yLevel) {
    if (levelSinceMs == 0) levelSinceMs = now;
    if (now - levelSinceMs >= HOLD_LEVEL_MS) stopPulses();
  }
}

void setup() {
  pinMode(MARKER_LED_PIN, OUTPUT);
  digitalWrite(MARKER_LED_PIN, LOW);
  Serial.begin(115200);
  WDT.begin(WATCHDOG_MS);
  lastCmdMs = lastLoopMs = millis();
}

void loop() {
  WDT.refresh();
  unsigned long now = millis();
  readSerial(now);

  if (armed && !ramping && now - lastCmdMs > TIMEOUT_MS) {
    ramping = true;
    levelSinceMs = 0;
  }
  if (ramping) rampToLevel(now);

  lastLoopMs = now;
}
