import sys
import time

import numpy as np
# noinspection PyPackageRequirements
from art.attacks.evasion import ZooAttack, \
    ProjectedGradientDescent as Pgd, HopSkipJump

from exp import VZoo, VPGD, VHSJ, AttackScore, LowProFool, \
    Validation, Validatable, cpgd_apply_and_predict, CPGD
from exp.utility import upper_attrs


class AttackPicker:
    """Lists all runnable attacks."""
    ZOO = 'zoo'
    HSJ = 'hsj'
    PDG = 'pgd'
    CPGD = 'cpgd'
    LPF = 'lpf'

    @staticmethod
    def list_attacks():
        return upper_attrs(AttackPicker)

    @staticmethod
    def load_attack(attack_name, apply_constr: bool):
        if attack_name == AttackPicker.ZOO:
            return VZoo if apply_constr else ZooAttack
        if attack_name == AttackPicker.PDG:
            return VPGD if apply_constr else Pgd
        if attack_name == AttackPicker.HSJ:
            return VHSJ if apply_constr else HopSkipJump
        if attack_name == AttackPicker.CPGD:
            return CPGD
        if attack_name == AttackPicker.LPF:
            return LowProFool


class AttackRunner:
    """Wrapper for running an adversarial attack"""

    def __init__(self, kind: str, constr: bool, conf):
        self.attack = AttackPicker.load_attack(kind, constr)
        self.name = self.attack.__name__
        self.constr = constr
        self.cls = None
        self.ori_x = None
        self.ori_y = None
        self.adv_x = None
        self.adv_y = None
        self.score = None
        self.conf = conf or {}
        self.start = self.end = 0

    def reset(self, cls):
        self.cls = cls
        self.score = AttackScore()
        self.ori_y = cls.test_y.copy()
        self.ori_x = cls.test_x.copy()
        self.adv_x = None
        self.adv_y = None
        self.start = self.end = 0
        return self

    @property
    def can_validate(self):
        return issubclass(self.attack, Validatable)

    def run(self, cge: Validation):
        """Generate adversarial examples and score."""
        self.start = time.time_ns()
        if issubclass(self.attack, CPGD):
            self.adv_x, self.adv_y = cpgd_apply_and_predict(
                self.cls.model, self.ori_x, self.ori_y, **self.conf)
        elif self.attack == LowProFool:
            feature_min = self.ori_x.min(0)[0]
            feature_max = self.ori_x.max(0)[0]
            feature_bounds = (feature_min, feature_max)
            # Initialize the LowProFool attack
            lpf_attack = LowProFool(
                classifier=self.cls.classifier,
                bounds=feature_bounds,
                **self.conf  
            )

            # Generate adversarial examples
            self.adv_x, original_preds, self.adv_y = lpf_attack.generate(
                x=self.ori_x,
                y=self.ori_y,
                feature_importance_method=self.conf.get('feature_importance_method', 'pearson')
            )
            # Preprocess ori_x and adv_x to align their shapes
            if len(self.ori_x.shape) == 1:  # If ori_x is 1D
                self.ori_x = self.ori_x[:, np.newaxis]  # Reshape to 2D (n_samples, 1)
            
            if len(self.adv_x.shape) == 1:  # If adv_x is 1D
                self.adv_x = self.adv_x[:, np.newaxis]  # Reshape to 2D (n_samples, 1)
            if self.ori_x.shape[1] != self.adv_x.shape[1]:  # Match feature dimensions
                max_features = max(self.ori_x.shape[1], self.adv_x.shape[1])
                self.ori_x = np.resize(self.ori_x, (self.ori_x.shape[0], max_features))
                self.adv_x = np.resize(self.adv_x, (self.adv_x.shape[0], max_features))
        else:
            aml_attack = self.attack(self.cls.classifier, **self.conf)
            if self.can_validate:
                aml_attack.vhost().cge = cge
            self.adv_x = aml_attack.generate(x=self.ori_x)
            self.adv_y = np.array(self.cls.predict(
                self.adv_x, self.ori_y).flatten())
        self.end = time.time_ns()
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')

        self.score.calculate(
            self, cge.constraints, cge.scalars,
            dur=self.end - self.start)
        return self

    def to_dict(self):
        return {
            'name': self.name,
            'config': self.conf,
            'can_validate': self.can_validate
        }
