"""Export models into different formats"""
import os
from pathlib import Path

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.train.serialization import export
from mindspore.train.serialization import load_checkpoint

from model_utils.moxing_adapter import config
from model_utils.moxing_adapter import moxing_wrapper
from src.dataset import get_number_of_entities_and_relations
from src.model_builder import create_model


def modelarts_pre_process():
    """modelarts pre process function."""
    config.file_name = os.path.join(config.output_path, config.file_name)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """Export model to the format specified by the user."""
    ent_tot, rel_tot = get_number_of_entities_and_relations(
        config.dataset_root,
        config.entities_file_name,
        config.relations_file_name,
    )
    net = create_model(ent_tot, rel_tot, config)

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)

    load_checkpoint(config.ckpt_file, net=net)
    net.set_train(False)

    hrt_data = Tensor(np.zeros([config.export_batch_size, 3], np.int32))

    export_dir = Path(config.export_output_dir)
    export_file = export_dir / config.model_name

    export(net, hrt_data, file_name=str(export_file), file_format=config.file_format)


if __name__ == '__main__':
    run_export()
