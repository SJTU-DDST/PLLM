from scripts.collect_route_mtp_traces import extract_question


def test_extract_question_from_multifield_prompt() -> None:
    prompt = (
        "The question is as follows: What is CUDA? "
        "The context is as follows: CUDA is a platform."
    )
    assert extract_question(prompt) == "What is CUDA?"


def test_extract_question_uses_last_few_shot_question() -> None:
    prompt = (
        "Question:\nExample?\nAnswer:\nExample answer\n"
        "Question:\nWhat is RDMA?\nAnswer:\n"
    )
    assert extract_question(prompt) == "What is RDMA?"
