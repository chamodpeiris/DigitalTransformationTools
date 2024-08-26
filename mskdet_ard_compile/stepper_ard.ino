#include <AccelStepper.h>

#define blue 9
#define pink 10
#define yellow 11
#define orange 12

#define HALFSTEP 8
#define FULLSTEP 4

AccelStepper stepper1(HALFSTEP, blue, yellow, pink, orange);

void setup()
{
    Serial.begin(9600);
    stepper1.setMaxSpeed(1000.0);
    stepper1.setAcceleration(100.0);
}

void loop()
{
    if (Serial.available() > 0)
    {
        // Read the input from serial
        String input = Serial.readStringUntil('\n');

        // Convert X coordinate to angle
        int x = input.toInt();
        int angle = map(x, 0, 600, -180, 180); // Assuming X coordinate ranges from 0 to 1200 pixels and mapping it to -180 to 180 degrees

        // Move the stepper motor
        stepper1.moveTo(angle);
    }

    stepper1.run();
}