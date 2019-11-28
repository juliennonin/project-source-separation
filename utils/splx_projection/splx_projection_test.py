import unittest
import numpy as np
import splx_projection

class SplxProjTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SplxProjTest, self).__init__(*args, **kwargs)
        self.x = splx_projection.splx_projection(np.random.rand(30,30), 1.)

    def test_sum_to_one(self):
        z = np.sum(self.x, axis=0)
        self.assertTrue(not (np.any(z < 1.-1e-10) 
                    and np.any(z > 1.+1e-10)
                    ))

    def test_non_negative(self):
        self.assertFalse(np.any(self.x < 0.))

if __name__ == '__main__':
  unittest.main()
