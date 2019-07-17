from unittest import TestCase

from robustness import main

from cox.utils import Parameters, override_json

class IntegrationTests(TestCase):
    def test_natural_train(self):
        args = Parameters({
            'train_mode':'nat',
            'dataset':'cifar'
        })

        args = override_json(args, 'robustness/configs/cifar10.json')
        args = override_json(args, 'robustness/configs/defaults.json')
        main.main(args)

    def test_robust_train(self, params):
        pass

    def test_linf_train(self):
        pass

    def test_l2_train(self):
        pass

if __name__ == '__main__':
    IntegrationTests().test_natural_train()
