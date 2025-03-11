# Health Is Wealth

## Inspiration
We were inspired to create Health is Wealth because we struggled with maintaining proper form during our workout sessions. We realized that without feedback on our technique, it was challenging to make progress and avoid injury.

## What it does
Health is Wealth utilizes an AI model powered by the mediapipe library to assist users during their workout sessions. It tracks exercise counts and movement using Python and OpenCV. By analyzing body part angles, the system determines the accuracy of each exercise repetition, providing real-time feedback to the user.

## How we built it
We built Health is Wealth using Python and OpenCV to capture and analyze live video streams of workout sessions. We integrated the mediapipe library to detect key body points and calculate relevant angles. The AI model then interprets this data to accurately count repetitions and evaluate the workout form.

## Challenges we ran into
One challenge we encountered was fine-tuning the AI model to accurately detect and interpret various exercise movements. Our list of supported workouts are push-ups, sit-ups, pull-ups, and walking. 
Additionally, achieving the real-time rep counting presented technical hurdles that we had to overcome.

## Accomplishments that we're proud of
We're proud to have developed a functional prototype of Health is Wealth that effectively assists users in maintaining proper form during workouts. In particular, we are very proud of our implementation of computer vision as this was our first full hackathon project to use computer vision and video analysis and it took a ton of brainpower among the two of us to flesh out the project and overcome hurdles fully.

## What we learned
Throughout the development process, we learned valuable lessons about computer vision, AI model integration, and real-time data processing. In particular, implementing the real-time workout recognition took a lot of time to figure out in terms of adjusting how to recognize if a rep is being successfully counted taking into account the camera angles that a user may provide and training the model for a clear cut way to track reps.

## What's next for Health is Wealth
In the future, we aim to further enhance the accuracy and versatility of Health is Wealth by incorporating additional exercises and refining the AI model's capabilities. We also plan to optimize the user interface for a seamless and intuitive experience, making it even easier for users to improve their workout performance and achieve their fitness goals.
