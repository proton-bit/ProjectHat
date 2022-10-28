# ProjectHat

### Progress

1. real-time Camera Calibration(Swift) 
    - [x] Write a filter 
    - [x] Estimate brightness of an image 
    - [x] integrate filter to given react app 

2. Collect info from photo
    - [ ] Blur (general defocus and specific people)
    - [x] Number of people 
    - [ ] Facial expressions/features
        - [ ] main approaches:
            - [ ] via yolo -> mediapipe
                - [ ] write mediapipe wrapper
                - [ ] compute features using position of landmarks

            - [ ] head-on solution
                - [ ] Closed eyes
                     - [ ] Download dataset:
                    https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset
                     - [ ] write pipeline
                     - [ ] train net
                - [ ] Smiles
                     - [ ] Download dataset:
                        link: None
                     - [ ] write pipeline
                     - [ ] train net
    5. - [ ] Write information into json sheet
        - [x] write parser 
        - [ ] collect annotations

3. Installing a hat. Maybe via special constructor
    - [x] Detect person 
    - [ ] Choose boxes contain people that we focus on
    - [x] Detect face landmarks 
    - [ ] compute center (x, y) coordinate of forehead
    - [ ] Install Hat
