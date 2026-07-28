from __future__ import annotations

import torch

from .backbones import load_dinov2
from .checkpoints import load_stage1_cost, load_stage2_checkpoint
from .concepts import ConceptBank
from .config import project_path, require_file
from .models.otf_cbm import OTFCBM


def build_stage2_model(
    config: dict,
    device: torch.device,
    checkpoint: str | None = None,
) -> OTFCBM:
    bank_path = require_file(config, config["concepts"]["bank"], "Concept bank")
    concept_bank = ConceptBank.load(bank_path)
    backbone = load_dinov2(config, device)
    num_classes = int(config["dataset"]["num_classes"])
    model = OTFCBM(backbone, concept_bank, num_classes, config)

    if checkpoint is None:
        stage1 = require_file(config, config["stage1_checkpoint"], "Stage-1 checkpoint")
        load_stage1_cost(model.cost_model, stage1)
        model.freeze_stage1_ot()
    else:
        checkpoint_path = project_path(config, checkpoint)
        assert checkpoint_path is not None
        load_stage2_checkpoint(model, checkpoint_path)
        model.freeze_stage1_ot()

    model.to(device)
    return model
