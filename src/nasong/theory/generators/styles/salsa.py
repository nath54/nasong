from nasong.theory.core.scale import Scale
from nasong.theory.structures.progression import Progression
from nasong.theory.systems.western import Western
from nasong.theory.structures.rhythm import Rhythm


class Salsa:
    """
    Generates Salsa patterns and montunos.
    """

    @staticmethod
    def montuno_progression(root: str = "G4", minor: bool = True) -> Progression:
        """
        Typical simple montuno: i - V
        """
        scale = Western.minor(root) if minor else Western.major(root)
        return Progression.from_roman_numerals(scale, ["i", "V"])

    @staticmethod
    def clave_rhythm(direction: str = "2-3"):
        """
        Returns a Clave rhythm pattern.
        """
        # 2-3 Son Clave: ..X.X... .X..X...
        # 3-2 Son Clave: X..X..X. ..X.X...
        pass
