import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_model(model_class_name, rand_seed=None, step_size=None):
    if model_class_name == 'ModelCNNMnist':
        from models.cnn_mnist import ModelCNNMnist
        return ModelCNNMnist()
    elif model_class_name == 'ModelCNNCifar10':
        from models.cnn_cifar10 import ModelCNNCifar10
        return ModelCNNCifar10()
    elif model_class_name == 'ModelSVMSmooth':
        from models.svm_smooth import ModelSVMSmooth
        return ModelSVMSmooth()
    else:
        raise Exception("Unknown model class name")
