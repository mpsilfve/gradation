from ..pkl_operations.pkl_io import *
from sklearn.preprocessing import normalize

from packages.constants.constants import *
from packages.utils.utils_iter import map_list


def normalize_embeds(frame):
    dims = map_list(str, ENCODER_DIMS)
    frame[dims] = normalize(frame[dims], axis=0)
    return frame

def obtain_penult_embed(rep_tensor):
    pen_index = len(rep_tensor) - 2
    pen_representation = rep_tensor[pen_index][0]
    assert len(pen_representation) == 500
    return pen_representation.numpy()

def build_validation_frame(validation_representations, validation_csv_path):
    validation_frame = pd.read_csv(validation_csv_path)
    validation_frame[ENCODER_DIMS] = list(map(obtain_penult_embed, validation_representations))
    store_csv_dynamic(validation_frame, 'gradation_w_encoder_state_validation')
    # TODO: add the dimensions to it.

def build_train_frame(train_representations, train_csv_path):
    train_frame = pd.read_csv(train_csv_path)
    train_frame[ENCODER_DIMS] = list(map(obtain_penult_embed, train_representations))
    store_csv_dynamic(train_frame, 'gradation_w_encoder_state_train')


def print_data_stats(train_frame, validation_frame):
    num_train_gradation = len(train_frame[train_frame[LABEL_COLUMN]=='yes'])
    num_train_no_gradation = len(train_frame[train_frame[LABEL_COLUMN]=='no'])

    num_test_gradation = len(validation_frame[validation_frame[LABEL_COLUMN]=='yes'])
    num_test_no_gradation = len(validation_frame[validation_frame[LABEL_COLUMN]=='no'])
    print(f"Number of train examples with gradation: {num_train_gradation}")
    print(f"Number of train examples without gradation: {num_train_no_gradation}")
    print(f"Proportion of train examples with gradation: {num_train_gradation/(num_train_no_gradation + num_train_gradation):.3f}")

    print(f"Number of test examples with gradation: {num_test_gradation}")
    print(f"Number of test examples without gradation: {num_test_no_gradation}")
    print(f"Proportion of test examples with gradation: {num_test_gradation/(num_test_gradation + num_test_no_gradation):.3f}")