"""Teacher modules: forward synthesis and back-translation."""

from agentsynth.teachers.back_translator import BackTranslator
from agentsynth.teachers.base_teacher import BaseTeacher

__all__ = ["BaseTeacher", "BackTranslator"]
