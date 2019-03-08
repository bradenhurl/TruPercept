from wavedata.tools.obj_detection import obj_utils, evaluation
from avod.core import box_3d_encoder

# Returns indices of objects
# This function is modified code from: https://github.com/kujason
def get_iou3d_matches(ego_objs, objs_perspectives):
    all_3d_ious = []

    # Convert to iou format
    ego_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
        ego_objs)
    perspect_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
        objs_perspectives)

    max_ious_3d = np.zeros(len(objs_perspectives))
    max_iou_pred_indices = -np.ones(len(objs_perspectives))
    for det_idx in range(len(objs_perspectives)):
        perspect_obj_iou_fmt = perspect_objs_iou_fmt[det_idx]

        ious_3d = evaluation.three_d_iou(perspect_obj_iou_fmt,
                                         ego_objs_iou_fmt)

        max_iou_3d = np.amax(ious_3d)
        max_ious_3d[det_idx] = max_iou_3d

        if max_iou_3d > 0.0:
            max_iou_pred_indices[det_idx] = np.argmax(ious_3d)

    return max_ious_3d, max_iou_pred_indices

def strip_objs(trust_objs):
    stripped_objs = []
    for trust_obj in trust_objs:
        stripped_objs.append(trust_obj.obj)

    return stripped_objs

# Returns a list of lists of objects which have been matched
def match_iou3ds(ego_trust_objs, perspectives_trust_objs):

    combined_trust_objs = []
    if perspectives_trust_objs != None:
        print("Here")
        for perspective_trust_objs in perspectives_trust_objs:
            if perspectives_trust_objs != None:
                print("Here2")
                for perspective_trust_obj in perspective_trust_objs:
                    combined_trust_objs.append(perspective_trust_obj)

    stripped_perspectives_trust_objs = strip_objs(combined_trust_objs)
    stripped_ego_trust_objs = strip_objs(ego_trust_objs)

    ego_max_ious, ego_iou_indices = get_iou3d_matches(stripped_ego_trust_objs, stripped_perspectives_trust_objs)

    print("Max_ious: ", ego_max_ious)
    print("Indices: ", ego_iou_indices)
    #TODO finish this
    return ego_iou_indices