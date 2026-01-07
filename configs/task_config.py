
# ----------------------------------------------------------
# GLUE META
# ----------------------------------------------------------
GLUE_META = {
    "mnli": dict(num_labels=3, main="accuracy",
                 s1="premise", s2="hypothesis", eval_key="validation_matched"),
    "sst2": dict(num_labels=2, main="accuracy",
                 s1="sentence", s2=None, eval_key="validation"),
    "cola": dict(num_labels=2, main="matthews_correlation",
                 s1="sentence", s2=None, eval_key="validation"),
    "mrpc": dict(num_labels=2, main="accuracy",
                 s1="sentence1", s2="sentence2", eval_key="validation"),
    "qqp": dict(num_labels=2, main="accuracy",
                s1="question1", s2="question2", eval_key="validation"),
    "qnli": dict(num_labels=2, main="accuracy",
                 s1="question", s2="sentence", eval_key="validation"),
    "rte": dict(num_labels=2, main="accuracy",
                s1="sentence1", s2="sentence2", eval_key="validation"),
    "stsb": dict(num_labels=1, main="pearson",
                 s1="sentence1", s2="sentence2", eval_key="validation"),
}




# PiSSA hyperparameters from Table 10
PISSA_TASK_CONFIG = {
    "mnli": dict(epochs=5, batch=16, lr=5e-4, alpha=8),
    "sst2": dict(epochs=20, batch=16, lr=3e-5, alpha=8),
    "mrpc": dict(epochs=20, batch=32, lr=2e-4, alpha=8),
    "cola": dict(epochs=20, batch=16, lr=1e-4, alpha=8),
    "qnli": dict(epochs=10, batch=32, lr=1e-4, alpha=16),
    "qqp": dict(epochs=10, batch=16, lr=1e-4, alpha=8),
    "rte": dict(epochs=50, batch=16, lr=1e-4, alpha=8),
    "stsb": dict(epochs=20, batch=8, lr=3e-4, alpha=8),
}

DORA_TASK_CONFIG = {
    "mnli": dict(epochs=10, batch=32, lr=2e-4, alpha=16),
    "sst2": dict(epochs=10, batch=16, lr=4e-4, alpha=16),
    "mrpc": dict(epochs=10, batch=32, lr=4e-4, alpha=16),
    "cola": dict(epochs=20, batch=8, lr=1e-4, alpha=6),
    "qnli": dict(epochs=10, batch=16, lr=2e-4, alpha=16),
    "qqp": dict(epochs=10, batch=16, lr=1e-4, alpha=6),
    "rte": dict(epochs=50, batch=8, lr=2e-4, alpha=6),
    "stsb": dict(epochs=20, batch=16, lr=3e-4, alpha=6),
}

LORA_TASK_CONFIG = {
    "mnli": dict(epochs=10, batch=32, lr=3e-4, alpha=8),
    "sst2": dict(epochs=10, batch=32, lr=1e-4, alpha=8),
    "mrpc": dict(epochs=10, batch=32, lr=1e-4, alpha=8),
    "cola": dict(epochs=30, batch=32, lr=4e-4, alpha=8),
    "qnli": dict(epochs=25, batch=32, lr=3e-4, alpha=8),
    "qqp": dict(epochs=10, batch=16, lr=3e-4, alpha=8),
    "rte": dict(epochs=50, batch=32, lr=4e-4, alpha=8),
    "stsb": dict(epochs=30, batch=16, lr=4e-4, alpha=8),
}

LAVA_TASK_CONFIG = {
    "mnli": dict(epochs=5, batch=8, lr=5e-4, alpha=8),
    "sst2": dict(epochs=20, batch=8, lr=3e-5, alpha=8),
    "mrpc": dict(epochs=20, batch=8, lr=2e-4, alpha=8),
    "cola": dict(epochs=20, batch=8, lr=1e-4, alpha=8),
    "qnli": dict(epochs=10, batch=16, lr=1e-4, alpha=16),
    "qqp": dict(epochs=10, batch=8, lr=1e-4, alpha=8),
    "rte": dict(epochs=50, batch=8, lr=1e-4, alpha=8),
    "stsb": dict(epochs=20, batch=8, lr=3e-4, alpha=8),
}


MOCA_TASK_CONFIG = LORA_TASK_CONFIG

# # Lava default (너가 원하는 걸 추가해도 됨)
# LAVA_TASK_CONFIG = {
#     task: dict(epochs=3, batch=32, lr=3e-4, alpha=8)
#     for task in GLUE_META.keys()
# }
