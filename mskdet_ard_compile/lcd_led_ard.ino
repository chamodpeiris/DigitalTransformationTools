#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); // Set the LCD address to 0x27 for a 16 chars and 2 line display
int ledPin = 4;                     // Define the pin for the LED

void setup()
{
    Serial.begin(9600);      // Initialize serial communication
    pinMode(ledPin, OUTPUT); // Set the LED pin as an output
    lcd.init();              // Initialize the LCD
    lcd.backlight();         // Turn on the backlight
}

void loop()
{
    if (Serial.available() > 0)
    {
        char state = Serial.read(); // Read data from serial port
        if (state == 'M')
        {                               // Mask detected
            lcd.clear();                // Clear the LCD
            lcd.setCursor(0, 0);        // Set cursor to first line
            lcd.print("Mask Detected"); // Print message on LCD
            digitalWrite(ledPin, LOW);  // Turn off the LED
        }
        else if (state == 'N')
        {                                  // No mask detected
            lcd.clear();                   // Clear the LCD
            lcd.setCursor(0, 0);           // Set cursor to first line
            lcd.print("No Mask Detected"); // Print message on LCD
            digitalWrite(ledPin, HIGH);    // Turn on the LED
        }
    }
}