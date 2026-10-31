#include <LiquidCrystal.h>
#include <string.h>

const uint8_t PUMP_PINS[] = {50, 51, 52, 53};
const uint8_t PUMP_COUNT = 4;
const unsigned long COMMAND_TIMEOUT_MS = 500UL;

LiquidCrystal lcd(12, 11, 5, 4, 3, 2);

char lineBuffer[12];
uint8_t lineLength = 0;
bool sawCarriageReturn = false;
bool discardingLine = false;
bool latchedOff = true;
bool watchdogArmed = false;
uint8_t activeMask = 0;
unsigned long lastValidCommandAt = 0;

void printPumpState(uint8_t pumpIndex) {
  lcd.print('P');
  lcd.print(pumpIndex + 1);
  lcd.print(':');
  lcd.print((activeMask & (1U << pumpIndex)) != 0 ? F("ON ") : F("OFF"));
}

void displayPumpStates(uint8_t mask) {
  activeMask = mask;

  lcd.setCursor(0, 0);
  printPumpState(0);
  lcd.print(' ');
  printPumpState(1);
  lcd.print(F("   "));

  lcd.setCursor(0, 1);
  printPumpState(2);
  lcd.print(' ');
  printPumpState(3);
  lcd.print(F("   "));
}

void applyPumpMask(uint8_t mask) {
  if (mask == activeMask) {
    return;
  }

  for (uint8_t i = 0; i < PUMP_COUNT; ++i) {
    const bool enabled = (mask & (1U << i)) != 0;
    digitalWrite(PUMP_PINS[i], enabled ? HIGH : LOW);
  }
  displayPumpStates(mask);
  Serial.print(F("EVENT STATE "));
  Serial.println(mask);
}

void latchOff() {
  applyPumpMask(0);
  latchedOff = true;
  watchdogArmed = false;
  displayPumpStates(0);
}

void rejectMalformed() {
  latchOff();
  Serial.println(F("EVENT REJECT MALFORMED"));
}

void rejectOverflow() {
  latchOff();
  Serial.println(F("EVENT REJECT OVERFLOW"));
}

bool parsePumpMask(const char *line, size_t length, uint8_t &mask) {
  static const char PREFIX[] = "PUMPS:";
  if (length < 7 || memcmp(line, PREFIX, sizeof(PREFIX) - 1) != 0) {
    return false;
  }

  if (length == 7 && line[6] >= '0' && line[6] <= '9') {
    mask = static_cast<uint8_t>(line[6] - '0');
    return true;
  }

  if (length == 8 && line[6] == '1' && line[7] >= '0' && line[7] <= '5') {
    mask = static_cast<uint8_t>(10 + line[7] - '0');
    return true;
  }

  return false;
}

void handleValidMask(uint8_t mask) {
  const unsigned long now = millis();
  if (latchedOff) {
    if (mask == 0) {
      latchedOff = false;
      lastValidCommandAt = now;
      watchdogArmed = true;
      displayPumpStates(0);
      Serial.println(F("EVENT REARMED"));
    } else {
      Serial.println(F("EVENT IGNORED LATCHED"));
    }
    return;
  }

  lastValidCommandAt = now;
  watchdogArmed = true;
  applyPumpMask(mask);
}

void resetInputLine() {
  lineLength = 0;
  sawCarriageReturn = false;
  discardingLine = false;
}

void discardUntilLineFeed() {
  lineLength = 0;
  sawCarriageReturn = false;
  discardingLine = true;
}

void finishInputLine() {
  uint8_t mask = 0;
  lineBuffer[lineLength] = '\0';
  if (parsePumpMask(lineBuffer, lineLength, mask)) {
    handleValidMask(mask);
  } else {
    rejectMalformed();
  }
  resetInputLine();
}

void consumeSerialByte(char byte) {
  if (discardingLine) {
    if (byte == '\n') {
      resetInputLine();
    }
    return;
  }

  if (byte == '\0') {
    rejectMalformed();
    discardUntilLineFeed();
    return;
  }

  if (byte == '\n') {
    finishInputLine();
    return;
  }

  if (byte == '\r') {
    if (sawCarriageReturn) {
      rejectMalformed();
      discardUntilLineFeed();
    } else {
      sawCarriageReturn = true;
    }
    return;
  }

  if (sawCarriageReturn) {
    rejectMalformed();
    discardUntilLineFeed();
    return;
  }

  if (lineLength >= sizeof(lineBuffer) - 1) {
    rejectOverflow();
    discardUntilLineFeed();
    return;
  }

  lineBuffer[lineLength++] = byte;
}

void enforceWatchdog() {
  if (!watchdogArmed) {
    return;
  }
  const unsigned long now = millis();
  if (now - lastValidCommandAt > COMMAND_TIMEOUT_MS) {
    latchOff();
    Serial.println(F("EVENT WATCHDOG"));
  }
}

void setup() {
  for (uint8_t i = 0; i < PUMP_COUNT; ++i) {
    pinMode(PUMP_PINS[i], OUTPUT);
    digitalWrite(PUMP_PINS[i], LOW);
  }

  lcd.begin(16, 2);
  displayPumpStates(0);

  Serial.begin(115200);
  Serial.println(F("MULTICAM_PUMP_READY"));
  Serial.println(F("EVENT BOOT"));
}

void loop() {
  while (Serial.available() > 0) {
    consumeSerialByte(static_cast<char>(Serial.read()));
  }
  enforceWatchdog();
}
