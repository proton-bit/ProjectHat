### TODO

1. real-time Camera Calibration(Swift) - Done ✅
    1. - [ ] Write a filter ✅
    2. - [ ] Estimate brightness of an image ✅
    3. - [ ] integrate filter to given react app ✅

2. Collect info from photo
    1. - [ ] Blur (general defocus and specific people)
    2. - [ ] Number of people ✅
    3. - [ ] Facial expressions/features
        2 - [ ] main approaches:
            1. - [ ] via yolo -> mediapipe
                1. w- [ ] rite mediapipe wrapper
                2. - [ ] compute features using position of landmarks

            2. - [ ] head-on solution
                1. - [ ] Closed eyes
                    1. - [ ] Download dataset:
                    https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset
                    2. - [ ] write pipeline
                    3. - [ ] train net
                2. Smiles
                    1. - [ ] Download dataset:
                        link: None
                    2. - [ ] write pipeline
                    3. - [ ] train net
    5. - [ ] Write information into json sheet
        1. - [ ] write parser ✅
        2. - [ ] collect annotations

3. Installing a hat. Maybe via special constructor
    1. - [ ] Detect person ✅
    2. - [ ] Choose boxes contain people that we focus on
    3. - [ ] Detect face landmarks ✅
    4. - [ ] compute center (x, y) coordinate of forehead
    5. - [ ] Install Hat
