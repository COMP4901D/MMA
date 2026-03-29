"""
    Defines the mapping from dataset activity labels to group labels for MuMu.
"""
UTD_ACTIVITY_TO_GROUP = {
    # Group 0: Arm/hand gestures
    0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 7:0, 8:0, 9:0, 10:0, 11:0, 17:0, 18:0, 19:0,
    # Group 1: Sports/ball actions
    6:1, 12:1, 13:1, 14:1, 15:1, 16:1,
    # Group 2: Upper body push/strength
    20:2, 21:2,
    # Group 3: Lower body/locomotion
    22:3, 23:3, 25:3, 26:3,
    # Group 4: Posture transitions
    24:4,
}

NUM_ACTIVITY_GROUPS = 5