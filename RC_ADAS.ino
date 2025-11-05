/*

 *  ADAS RC CAR (UNO R4) – Sensor Fusion, Obstacle Avoidance & BLE Control
 *  Description:
 *  This sketch implements a simplified Advanced Driver Assistance System (ADAS)
 *  using ultrasonic and radar sensors for obstacle detection and avoidance.
 *  The system supports manual control via Bluetooth (HM-10) and autonomous modes
 *  that respond to front/side ultrasonic and RD-03D radar data.
 *
 *  Hardware:
 *    - Arduino UNO R4 Minima
 *    - 3x HC-SR04 ultrasonic sensors (front, left, right)
 *    - RD-03D 60° radar module
 *    - HM-10 Bluetooth module
 *    - Standard servo for steering
 *    - L298N motor driver (PWM on ENA, IN1/IN2 for direction)
 *
 *  Modes:
 *    0 – OFF
 *    1 – Ultrasonic only
 *    2 – Radar only
 *    3 – Combined (fusion mode)
 *
 *  Author: Imran Tauqeer (Mechatronic Engineering, USYD)
 *  Last modified: 5 Nov 2025
 * 
 */

#include <Arduino.h>
#include <Servo.h>
#include <RadarSensor.h>
#include <math.h>


// Pin Definitions

const uint8_t ENA        = 5;     // Motor PWM (throttle)
const uint8_t IN1        = 6;     // Motor direction A
const uint8_t IN2        = 7;     // Motor direction B
const uint8_t SERVO_PIN  = 10;    // Steering servo signal

const uint8_t TRIG_PIN   = A0;    // Shared trigger for all ultrasonic sensors
const uint8_t FRONT_ECHO = A3;    // Front ultrasonic echo
const uint8_t RIGHT_ECHO = A1;    // Right ultrasonic echo
const uint8_t LEFT_ECHO  = A2;    // Left ultrasonic echo

// Radar (RD-03D): TX → D8 (Arduino RX), RX unused
RadarSensor radar(8, 9);
const unsigned long RADAR_BAUD = 256000;

// HM-10 BLE module on hardware UART
const unsigned long BLE_BAUD = 9600;


// Drive & Servo Constants

const int SERVO_US_CENTER = 1500;
const int SERVO_US_LEFT   = 1900;
const int SERVO_US_RIGHT  = 1100;

const uint8_t SPEED_FWD_RC   = 255;   // Full forward (manual)
const uint8_t SPEED_REV_RC   = 255;
const uint8_t SPEED_ADAS_FWD = 200;   // Reduced forward during ADAS manoeuvres

// Steering configuration
const bool ANGLE_POS_IS_LEFT = false; // Flip if radar reports inverted angles
const bool INVERT_SERVO_DIR  = true;  // Reverse steering direction if linkage inverted

// Radar steering behaviour
const float ANGLE_DEADBAND_DEG = 1.0f;
const float SIGN_HYST_DEG      = 4.0f;
const float CENTER_LOCK_DEG    = 8.0f;
const float STEER_MAX_FRAC     = 1.0f;

// Steer-back control (to prevent over-steering or oscillation)
const float STEERBACK_ANGLE_DEG      = 30.0f;
const float STEERBACK_RELEASE_DEG    = 18.0f;
const float STEERBACK_K              = 0.6f;
const float STEERBACK_RATE_US_PER_MS = 120.0f;
const unsigned long STEERBACK_MIN_HOLD_MS = 600;
const unsigned long RADAR_MAX_AWAY_MS     = 1200;

// Ultrasonic failsafe (always active)
const float ULTRA_FAILSAFE_CM = 25.0f;

// Smoothing parameters
const float LPF_ALPHA_ANGLE    = 0.8f;
const float LPF_ALPHA_STEER_US = 0.8f;
const float SLEW_US_PER_MS     = 40.0f;
const float ANGLE_JUMP_CLAMP   = 90.0f;
const float ANGLE_SHAPE_GAMMA  = 1.0f;
const float MIN_K_AWAY         = 0.3f;
const unsigned long RADAR_STICKY_MS = 500;

// Joystick override control
const float JOY_OVERRIDE_INTENSITY = 0.3f;
const unsigned long JOY_SUPPRESS_RADAR_MS = 500;

// Ultrasonic behaviour thresholds
const float FRONT_TRIGGER_CM = 60.0f;
const float SIDE_BLOCK_CM    = 20.0f;
const float SIDE_TIE_EPS     = 6.0f;

const unsigned long TURN_AWAY_MS   = 1000;
const unsigned long STRAIGHTEN_MS  = 800;
const unsigned long CENTER_SLAM_MS = 100;
const unsigned long REFRACT_MS     = 600;

// Radar detection window
const float RADAR_TRIGGER_CM = 300.0f;
const float RADAR_ANGLE_MIN  = -60.0f;
const float RADAR_ANGLE_MAX  = 60.0f;


// Data Structures & Global Variables

struct Med3 { float a=9999, b=9999, c=9999; };
struct RadarQuick { bool has=false; float dist_cm=9999, angle_deg=0, speed_cms=0; };
struct A3 { float a=0,b=0,c=0; } a3;

Servo steer;
Med3 mfF, mfL, mfR;

enum DodgeDir   { DD_LEFT, DD_RIGHT };
enum AvState    { AV_CRUISE, AV_TURN_AWAY, AV_STRAIGHTEN, AV_CENTER_HOLD };
enum DriveState { DRIVE_STOP, DRIVE_FWD, DRIVE_REV };
enum AdasMode   { ADAS_OFF=0, ADAS_ULTRASONIC_ONLY=1, ADAS_RADAR_ONLY=2, ADAS_BOTH=3 };

AdasMode adasMode = ADAS_ULTRASONIC_ONLY;


// Utility Functions

const char* modeName(AdasMode m){
  switch(m){
    case ADAS_OFF: return "OFF";
    case ADAS_ULTRASONIC_ONLY: return "ULTRASONIC_ONLY";
    case ADAS_RADAR_ONLY: return "RADAR_ONLY";
    case ADAS_BOTH: return "BOTH";
  }
  return "?";
}

static inline float clamp01(float x){ return x<0?0:(x>1?1:x); }


// Motor Helpers

void motorStop()              { analogWrite(ENA, 0); digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); }
void motorForward(uint8_t p)  { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  analogWrite(ENA, p); }
void motorReverse(uint8_t p)  { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); analogWrite(ENA, p); }


// Servo Helpers

static inline int targetRightUS(){ return INVERT_SERVO_DIR ? SERVO_US_LEFT  : SERVO_US_RIGHT; }
static inline int targetLeftUS (){ return INVERT_SERVO_DIR ? SERVO_US_RIGHT : SERVO_US_LEFT;  }

void setCenter(){ steer.writeMicroseconds(SERVO_US_CENTER); }
void setLeft()  { steer.writeMicroseconds(targetLeftUS()); }
void setRight() { steer.writeMicroseconds(targetRightUS()); }


// Ultrasonic Measurement (median filter on 3 samples)

static float med3(float x, Med3 &m){
  m.c=m.b; m.b=m.a; m.a=x;
  float x1=m.a,x2=m.b,x3=m.c;
  if(x1>x2) swap(x1,x2);
  if(x2>x3) swap(x2,x3);
  if(x1>x2) swap(x1,x2);
  return x2;
}

static float readUltraCM(uint8_t trig, uint8_t echo, unsigned long tout=30000UL){
  digitalWrite(trig,LOW); delayMicroseconds(2);
  digitalWrite(trig,HIGH); delayMicroseconds(10);
  digitalWrite(trig,LOW);
  unsigned long d = pulseIn(echo, HIGH, tout);
  if(!d) return 9999.f;
  return d * 0.0343f * 0.5f;
}

static float frontCM(){ return med3(readUltraCM(TRIG_PIN,FRONT_ECHO), mfF); }
static float leftCM (){ return med3(readUltraCM(TRIG_PIN,LEFT_ECHO ), mfL); }
static float rightCM(){ return med3(readUltraCM(TRIG_PIN,RIGHT_ECHO), mfR); }


// Radar Measurement and Gating

static RadarQuick pollRadar(){
  RadarQuick rq;
  if (radar.update()){
    RadarTarget t = radar.getTarget();
    rq.has       = true;
    rq.dist_cm   = t.distance / 10.0f;
    rq.angle_deg = t.angle;
    rq.speed_cms = t.speed;
  }
  return rq;
}

static inline bool radarInGate(float ang, float dist){
  return (ang >= RADAR_ANGLE_MIN && ang <= RADAR_ANGLE_MAX) &&
         (dist > 0 && dist <= RADAR_TRIGGER_CM);
}

// angle stabiliser 
static inline float med3_angle(float x){
  a3.c=a3.b; a3.b=a3.a; a3.a=x;
  float x1=a3.a,x2=a3.b,x3=a3.c;
  if (x1>x2){ float t=x1;x1=x2;x2=t; }
  if (x2>x3){ float t=x2;x2=x3;x3=t; }
  if (x1>x2){ float t=x1;x1=x2;x2=t; }
  return x2;
}

static int lastAwaySign = +1; // +1 = steer RIGHT (away from person on left)
static inline int mapAwayUS(int awaySign, float k){
  if (k < MIN_K_AWAY) k = MIN_K_AWAY;     // never weaker than this
  if (k > STEER_MAX_FRAC) k = STEER_MAX_FRAC;
  int target = (awaySign > 0) ? targetRightUS() : targetLeftUS();
  return (int)(SERVO_US_CENTER + (target - SERVO_US_CENTER) * k);
}

int radar_applySteer(float angleDeg_raw, float dist_cm){
  if (!filtAngleInit){ filtAngleDeg = ANGLE_POS_IS_LEFT ? angleDeg_raw : -angleDeg_raw; filtAngleInit = true; }

  // Convert sensor convention: + = left
  float a = ANGLE_POS_IS_LEFT ? angleDeg_raw : -angleDeg_raw;

  // Median + jump clamp + LPF
  float a_med = med3_angle(a);
  if (fabs(a_med - filtAngleDeg) > ANGLE_JUMP_CLAMP){
    a_med = (a_med > filtAngleDeg) ? (filtAngleDeg + ANGLE_JUMP_CLAMP)
                                   : (filtAngleDeg - ANGLE_JUMP_CLAMP);
  }
  filtAngleDeg = (1.0f - LPF_ALPHA_ANGLE) * filtAngleDeg + LPF_ALPHA_ANGLE * a_med;

  // Determine away side with hysteresis
  int awaySignPrev = lastAwaySign;
  if (fabs(filtAngleDeg) > SIGN_HYST_DEG){
    lastAwaySign = (filtAngleDeg >= 0) ? +1 : -1; // +angle => person LEFT => steer RIGHT
  }
  float aabs = fabs(filtAngleDeg);
  if (aabs < ANGLE_DEADBAND_DEG) aabs = 0.0f;

  unsigned long now = millis();

  // Track how long have we been steering AWAY
  if (lastAwaySign != awaySignPrev || aabs < ANGLE_DEADBAND_DEG){
    awaySinceMs = now; // reset timer if side flips or we are near center
  }

  //ENTER STEER-BACK if target is far to the side and drifting away 
  float dAbs = aabs - lastAngleAbs;     // angle magnitude trend
  lastAngleAbs = aabs;
  if (!inBackSteer){
    bool sideLarge   = (aabs >= STEERBACK_ANGLE_DEG);
    bool movingWorse = (dAbs > 0.4f);   // getting more to the side (tune 0.2~1.0)
    bool dwellTooLong= (now - awaySinceMs) > RADAR_MAX_AWAY_MS;

    if ( (sideLarge && movingWorse) || dwellTooLong ){
      inBackSteer = true;
      backSteerUntil = now + STEERBACK_MIN_HOLD_MS;
      LOG_EV("RADAR_STEERBACK_ON");
      // fall-through to steer-back block below
    }
  }

  // If in back-steer, drive opposite to AWAY side until released
  if (inBackSteer){
    steerTowardSign(-lastAwaySign, STEERBACK_K, STEERBACK_RATE_US_PER_MS);

    if ( (aabs <= STEERBACK_RELEASE_DEG && now >= backSteerUntil) || (aabs < ANGLE_DEADBAND_DEG) ){
      inBackSteer = false;
      awaySinceMs = now; // reset dwell window
      LOG_EV("RADAR_STEERBACK_OFF");
    }
    return lastServoUS; // while back-steering, skip normal 'away' logic
  }

  //Normal "steer away" authority
  float toward = 1.0f - clamp01(aabs / 60.0f);
  float toward_shaped = powf(toward, ANGLE_SHAPE_GAMMA);
  float d_norm = 1.0f - clamp01(dist_cm / RADAR_TRIGGER_CM);

  float k = 0.30f + 0.60f * toward_shaped + 0.30f * d_norm;
  if (aabs <= CENTER_LOCK_DEG) k = 1.0f;
  if (k > STEER_MAX_FRAC) k = STEER_MAX_FRAC;

  int rawUS = mapAwayUS(lastAwaySign, k);

  // Servo LPF + slew
  unsigned long dt = now - lastServoTS; if (dt == 0) dt = 1;
  int lpfUS = (int)((1.0f - LPF_ALPHA_STEER_US) * lastServoUS + LPF_ALPHA_STEER_US * rawUS);
  int maxStep = (int)(SLEW_US_PER_MS * dt);
  int delta = lpfUS - lastServoUS;
  if (delta >  maxStep) delta =  maxStep;
  if (delta < -maxStep) delta = -maxStep;
  int usCmd = lastServoUS + delta;

  steer.writeMicroseconds(usCmd);
  lastServoUS = usCmd; lastServoTS = now;
  return usCmd;
}

// Decisions (ultrasonic) 
static inline bool leftBlocked (float dl){ return dl <= SIDE_BLOCK_CM; }
static inline bool rightBlocked(float dr){ return dr <= SIDE_BLOCK_CM; }
static inline bool bothSidesBlocked(float dl, float dr){ return leftBlocked(dl) && rightBlocked(dr); }

static DodgeDir chooseDir(float dl, float dr){
  // 1) Hard rules when one side is blocked
  bool Lb = (dl <= SIDE_BLOCK_CM);
  bool Rb = (dr <= SIDE_BLOCK_CM);

  if ( Lb && !Rb) return DD_RIGHT;   // left blocked -> go right
  if (!Lb &&  Rb) return DD_LEFT;    // right blocked -> go left

  // 2) If both blocked, alternate to avoid deadlock
  if (Lb && Rb)  return (lastDir==DD_LEFT) ? DD_RIGHT : DD_LEFT;

  // 3) Both clear: go to the side with more clearance (with tie band)
  const float TIE = 8.0f; // cm
  float diff = dr - dl;   // + means right is clearer
  if (diff >  TIE) return DD_RIGHT;
  if (diff < -TIE) return DD_LEFT;

  // 4) If nearly equal, alternate for variety
  return (lastDir==DD_LEFT) ? DD_RIGHT : DD_LEFT;
}

//BLE / USB parsing 
String lineUSB, lineBT;

void cycleAdasMode(){
  adasMode = static_cast<AdasMode>((static_cast<int>(adasMode) + 1) % 4); // 4 modes
  Serial.print("[BTN] B0=ADAS mode -> ");
  Serial.println(modeName(adasMode));
  LOG_EV_F("MODE_CHANGE", modeCurrentForLog());
}

void doDriveFromButton(char b){
  // B3=brake/stop, B2=reverse latch, B0=cycle ADAS mode, B1=forward latch
  if (b=='3'){ driveState = DRIVE_STOP; motorStop(); Serial.println("[BTN] B3=Brake -> STOP"); LOG_EV("DRIVE_STOP"); }
  else if (b=='2'){ driveState = DRIVE_REV;  Serial.println("[BTN] B2=Reverse LATCH"); LOG_EV("DRIVE_REV"); }
  else if (b=='0'){ cycleAdasMode(); }
  else if (b=='1'){ driveState = DRIVE_FWD;  Serial.println("[BTN] B1=Forward LATCH"); LOG_EV("DRIVE_FWD"); }
}

void parseControlLine(const char* s){
  if ((s[0]=='B' || s[0]=='b') && s[1]>='0' && s[1]<='9'){ doDriveFromButton(s[1]); return; }

  if ((s[0]=='J' || s[0]=='j') && s[1]=='0'){
    // J0:<angle>,<intensity>
    const char* p = s+2;
    while (*p==' '||*p==':') p++;
    char* endp=nullptr;
    long ang = strtol(p,&endp,10); if (endp==p) return;
    p=endp;
    while (*p==' '||*p==','||*p==':') p++;
    endp=nullptr;
    double inten = strtod(p,&endp); if (endp==p) return;

    joyAngleDeg = (float)((ang%360+360)%360);
    joyIntensity = (float)inten;

    if (joyIntensity > JOY_OVERRIDE_INTENSITY){
      const char* lbl="CENTER";
      int us = steerUS_fromJoystick(joyAngleDeg, joyIntensity, &lbl);
      steer.writeMicroseconds(us);
      lastServoUS = us; lastServoTS = millis();
      joySuppressRadarUntil = millis() + JOY_SUPPRESS_RADAR_MS;
      Serial.print("[JOY-OVRD] "); Serial.print(lbl); Serial.print(" us="); Serial.println(us);
      LOG_EV("JOY_OVERRIDE");
    } else if (avState==AV_CRUISE){
      const char* lbl="CENTER";
      int us = steerUS_fromJoystick(joyAngleDeg, joyIntensity, &lbl);
      steer.writeMicroseconds(us);
      lastServoUS = us; lastServoTS = millis();
      Serial.print("[JOY] "); Serial.print(lbl); Serial.print(" us="); Serial.println(us);
    }
  }
}

void pumpSerial(HardwareSerial& port, String& buf){
  while (port.available()){
    char c = (char)port.read();
    if (c=='\r') continue;
    if (c=='\n'){
      if (buf.length()){
        parseControlLine(buf.c_str());
        buf.remove(0);
      }
    } else {
      if (buf.length() < 64) buf += c;
    }
  }
}

//Setup
void setup(){
  Serial.begin(115200);       // USB (Arduino IDE / Processing)
  Serial1.begin(BLE_BAUD);    // HM-10

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  motorStop();

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(FRONT_ECHO, INPUT);
  pinMode(LEFT_ECHO,  INPUT);
  pinMode(RIGHT_ECHO, INPUT);

  steer.attach(SERVO_PIN);
  setCenter();

  radar.begin(RADAR_BAUD);
  lastServoTS = millis();

  Serial.println("UNO R4 ADAS: Modes = OFF / ULTRASONIC_ONLY / RADAR_ONLY / BOTH (B0 cycles).");
  Serial.println("Ultrasonic has priority: while a US maneuver runs, radar steering is paused.");
  Serial.println("Controls: J0:<ang>,<int>; B3 stop; B2 reverse; B0 ADAS mode; B1 forward.");

  LOG_EV("TRIAL_START");
  LOG_EV_F("MODE_CHANGE", modeCurrentForLog());
}

//Loop 
void loop(){
  // Accept control from BOTH USB (Serial) and BT (Serial1)
  pumpSerial(Serial,  lineUSB);
  pumpSerial(Serial1, lineBT);

  // Sensor reads 
  float df = frontCM();
  float dl = leftCM();
  float dr = rightCM();

  RadarQuick rq = pollRadar();
  unsigned long now = millis();

  // Radar gate edges -> to measure reaction time windows in post
  bool radarGateNow = rq.has && radarInGate(rq.angle_deg, rq.dist_cm);
  if (radarGateNow && !radarGatePrev) { LOG_EV_F("RADAR_GATE_ON", rq.dist_cm); }
  if (!radarGateNow && radarGatePrev) { LOG_EV("RADAR_GATE_OFF"); }
  radarGatePrev = radarGateNow;

  // Optional: compact radar line
  if (rq.has){
    long dist_mm = (long)(rq.dist_cm * 10.0f);
    Serial.print("R,");
    Serial.print(rq.angle_deg, 1);  Serial.print(",");
    Serial.print(dist_mm);          Serial.print(",");
    Serial.println(rq.speed_cms, 1);
  }

  // Front ultrasonic failsafe (independent of ADAS mode) 
  static bool fsActive = false;
  static unsigned long fsTime = 0;

  float df_now = frontCM(); // read once for failsafe check
  if (!fsActive && df_now <= ULTRA_FAILSAFE_CM) {
    // Trigger once: hard stop and hold
    motorStop();
    setCenter();
    driveState = DRIVE_STOP;
    fsActive = true;
    fsTime = millis();
    Serial.println("[FAILSAFE] Triggered -> Hard Stop (1s hold)");
  }

  // After 1 second, allow normal control again
  if (fsActive && (millis() - fsTime >= 1000)) {
    fsActive = false;
    Serial.println("[FAILSAFE] Released -> Normal control restored");
  }

  // Throttle 
  switch (driveState){
    case DRIVE_STOP: motorStop(); break;
    case DRIVE_REV:  motorReverse(SPEED_REV_RC); break;
    case DRIVE_FWD: {
      // Slow only while ultrasonic sequence active
      uint8_t sp = (avState!=AV_CRUISE) ? SPEED_ADAS_FWD : SPEED_FWD_RC;
      motorForward(sp);
    } break;
  }

  //Mode gates
  const bool ULTRA_ALLOWED = (adasMode == ADAS_ULTRASONIC_ONLY || adasMode == ADAS_BOTH);
  const bool RADAR_ALLOWED = (adasMode == ADAS_RADAR_ONLY      || adasMode == ADAS_BOTH);

  // ULTRASONIC avoidance state machine 
  if (avState != AV_CRUISE){
    // Ultrasonic has priority: while active, we DO NOT run radar steering
    if (avState == AV_TURN_AWAY){
      if (bothSidesBlocked(dl, dr)){
        driveState = DRIVE_STOP; motorStop(); setCenter();
        avState = AV_CRUISE; refractUntil = now + REFRACT_MS;
        LOG_EV("US_BOTH_BLOCKED_STOP");
      } else if (now - avT0 >= TURN_AWAY_MS){
        // COUNTER-STEER: opposite to initial swerve
        if (currentDir == DD_LEFT){ setRight(); }
        else                       { setLeft();  }
        avT0 = now; avState = AV_STRAIGHTEN;
        LOG_EV("START_US_STRAIGHTEN");
      }
    } else if (avState == AV_STRAIGHTEN){
      if (bothSidesBlocked(dl, dr)){
        driveState = DRIVE_STOP; motorStop(); setCenter();
        avState = AV_CRUISE; refractUntil = now + REFRACT_MS;
        LOG_EV("US_BOTH_BLOCKED_STOP");
      } else if (now - avT0 >= STRAIGHTEN_MS){
        setCenter(); avT0 = now; avState = AV_CENTER_HOLD;
        LOG_EV("START_US_CENTER_HOLD");
      }
    } else if (avState == AV_CENTER_HOLD){
      if (now - avT0 >= CENTER_SLAM_MS){
        avState = AV_CRUISE; refractUntil = now + REFRACT_MS;
        LOG_EV("END_US_SEQUENCE");
      }
    }
  } else {
    // Not in ultrasonic sequence: may start one (if allowed), and/or run radar (if allowed).
    if (!fsActive && now >= refractUntil && ULTRA_ALLOWED && df < FRONT_TRIGGER_CM){
      if (bothSidesBlocked(dl, dr)){
        driveState = DRIVE_STOP; motorStop(); setCenter();
        refractUntil = now + REFRACT_MS;
        LOG_EV("US_BOTH_BLOCKED_STOP");
      } else {
        currentDir = chooseDir(dl, dr);
        lastDir = currentDir;
        // Initial swerve
        if (currentDir == DD_LEFT){ setLeft(); } else { setRight(); }
        avT0 = now; avState = AV_TURN_AWAY;
        LOG_EV_F("START_US_TURN", (currentDir==DD_LEFT ? -1.0f : +1.0f)); // -1=left, +1=right swerve
        // NOTE: from here until AV_CRUISE again, radar steering is paused by the avState!=AV_CRUISE block
      }
    }

    // LIVE RADAR steering (only when not in ultrasonic AND allowed by mode) 
    bool joySuppress = (now < joySuppressRadarUntil);
    if (!fsActive && RADAR_ALLOWED && !joySuppress && joyIntensity <= JOY_OVERRIDE_INTENSITY){
      bool gate = rq.has && radarInGate(rq.angle_deg, rq.dist_cm);
      if (gate) radarLastSeenMs = now;

      bool sticky = (now - radarLastSeenMs) <= RADAR_STICKY_MS;

      if (gate){
        (void)radar_applySteer(rq.angle_deg, rq.dist_cm);
      } else if (sticky){
        float holdDist = RADAR_TRIGGER_CM * 0.75f; // pretend “still close”
        (void)radar_applySteer( ANGLE_POS_IS_LEFT ? +fabs(filtAngleDeg) * (lastAwaySign<0?-1:1)
                                                  : -fabs(filtAngleDeg) * (lastAwaySign<0?-1:1),
                                holdDist);
      } else {
        // gently decay to center when no target
        unsigned long dt = now - lastServoTS; if (dt == 0) dt = 1;
        int target = SERVO_US_CENTER;
        int maxStep = (int)(SLEW_US_PER_MS * dt);
        int delta = target - lastServoUS;
        if (delta >  maxStep) delta =  maxStep;
        if (delta < -maxStep) delta = -maxStep;
        int usCmd = lastServoUS + delta;
        steer.writeMicroseconds(usCmd);
        lastServoUS = usCmd; lastServoTS = now;
      }
    }
  }

  //Periodic results line (10 Hz) 
  static unsigned long lastLog=0;
  if (millis() - lastLog >= 100) {
    logStatus(rq, df, dl, dr, driveState, avState, lastServoUS, joyIntensity);
    lastLog = millis();
  }

  //RAW DATA LOGGER (every 100 ms)
  static unsigned long lastRawLog = 0;
  if (millis() - lastRawLog >= 100) {
    lastRawLog = millis();

    // Raw ultrasonic pulse readings
    float df_raw = readUltraCM(TRIG_PIN, FRONT_ECHO);
    float dl_raw = readUltraCM(TRIG_PIN, LEFT_ECHO);
    float dr_raw = readUltraCM(TRIG_PIN, RIGHT_ECHO);

    // Latest radar reading (if any)
    RadarQuick rq_raw = pollRadar();

    // Print a CSV-style line with all values
    Serial.print("RAW,");
    Serial.print(millis()); Serial.print(',');     // timestamp
    Serial.print(modeName(adasMode)); Serial.print(','); // ADAS mode text
    Serial.print(df_raw,1); Serial.print(',');     // front ultrasonic (cm)
    Serial.print(dl_raw,1); Serial.print(',');     // left ultrasonic (cm)
    Serial.print(dr_raw,1); Serial.print(',');     // right ultrasonic (cm)
    Serial.print(rq_raw.has ? 1 : 0); Serial.print(',');  // radar valid flag
    Serial.print(rq_raw.angle_deg,1); Serial.print(',');  // radar angle (deg)
    Serial.print(rq_raw.dist_cm,1); Serial.print(',');    // radar distance (cm)
    Serial.print(rq_raw.speed_cms,1); Serial.print(',');  // radar speed (cm/s)
    Serial.print(lastServoUS); Serial.print(',');         // servo pulse (µs)
    Serial.print(driveState); Serial.print(',');          // drive state numeric
    Serial.println(avState);                              // ultrasonic state numeric
  }
}
