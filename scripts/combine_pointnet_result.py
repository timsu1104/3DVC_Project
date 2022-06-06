import json, os, argparse, glob

NUM_OBJECTS = 79

testing_data_dir = "datasets/testing_data/data"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', help='', default='')
    parser.add_argument('-o', help='output directory', default='outputs/PointNet.json')
    opt = parser.parse_args()
    TAG = opt.tag
    result_path = opt.o
    exp_path = os.path.join('exp/PointNet', TAG)

    rgb_files = sorted(glob.glob(os.path.join(testing_data_dir, '*_color_kinect.png')))
    scene_name = [rgb.split('/')[-1].split('_')[0] for rgb in rgb_files]

    preds = {}
    # Initialize
    for j, scene in enumerate(scene_name):
        pred_of_scene = ['null' for _ in range(NUM_OBJECTS)]
        preds[scene] = {"poses_world": pred_of_scene}

    # Filling in
    for label in range(NUM_OBJECTS):
        result_file = os.path.join(exp_path, str(label), 'test_result.json')
        if os.path.exists(result_file) == False:
            continue
        with open(result_file, "r") as f:
            pred = json.load(f)
        # structure of pred: {scene_name -> pred_pose}
        # structure of preds: {scene_name -> poses_world -> label -> pred_pose}
        for scene, pose in pred.items():
            preds[scene]["poses_world"][label] = pose

    with open(result_path, 'w') as f:
        json.dump(preds, f)