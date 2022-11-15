# ProjectHat

### Progress

1. real-time Camera Calibration(Swift) 
    - [x] Write a filter 
    - [x] Estimate brightness of an image 
    - [x] integrate filter to given react app 

2. Collect info from photo
    - [x] Choose boxes contain people that we focus on
    - [ ] Detect blurred images (general defocus and specific people)
    - [x] Number of people 
    - [x] Facial expressions/features
        - [x] yolo -> mediapipe
            - [x] write mediapipe wrapper
            - [x] localize head
            - [x] apply NMS
            - [x] compute features using position of landmarks
       - [x] Write information into json sheet
       - [x] write parser 
       - [x] collect annotations

3. Installing a hat. Maybe via special constructor
    - [x] Detect person 
    - [x] Choose boxes contain people that we focus on
    - [x] Detect face landmarks 
    - [ ] compute center (x, y) coordinate of forehead
    - [ ] adjust the mask to the position of the head
    - [ ] Install Hat
