import torch
import torch.nn.functional as F


def knowledge_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    KD is performed only on old classes.

    The teacher has outputs for old classes.
    The student has outputs for old + new classes.
    Therefore, student logits are sliced to match teacher logits.
    """
    old_num_classes = teacher_logits.shape[1]
    student_old_logits = student_logits[:, :old_num_classes]

    log_p_student = F.log_softmax(student_old_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)

    kd_loss = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="batchmean",
    ) * (temperature ** 2)

    return kd_loss
