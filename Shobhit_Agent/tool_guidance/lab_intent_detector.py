def detect_lab_intent(question):

    q = question.lower()

    # Explicit lab/tutorial intent
    if any(k in q for k in [
        "lab",
        "tutorial",
        "step by step",
        "hands on",
        "practice",
        "implementation",
        "walkthrough"
    ]):
        return True

    # Build/design intent
    if any(k in q for k in [
        "build",
        "create",
        "design",
        "implement",
        "deploy",
        "setup",
        "configure"
    ]):
        return True

    # Learning intent
    if any(k in q for k in [
        "how to",
        "learn",
        "example",
        "guide"
    ]):
        return True

    return False
