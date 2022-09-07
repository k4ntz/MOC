import pandas as pd

IMG_SIZE = 128

def _to_dataframe(bbs):
    """
    y_min, y_max, x_min, x_max format
    """
    return pd.DataFrame.from_dict(
        {i: [bb[0] / 210, bb[0] / 210 + bb[2] / 210, bb[1] / 160, bb[1] / 160 + bb[3] / 160, bb[4], bb[5]]
         for i, bb in enumerate(bbs)}, orient='index'
    )


def save(args, frame, info, output_path, visualizations):
    bb = _to_dataframe(info['bbs'])
    if bb.empty:
        raise RuntimeError("no gt bounding boxes found. Exiting with error...")
    bb = bb[(bb[0] >= 0) & (bb[0] <= 128) & (bb[1] >= 0) & (bb[1] <= 128)]
    bb.to_csv(output_path, header=False, index=False)
    for vis in visualizations:
        vis.save_vis(frame, bb)
