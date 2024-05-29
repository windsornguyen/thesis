import torch
import torch.nn as nn


class STU(nn.Module):
    def __init__(
        self: nn.Module,
        d_out=256: int,
        input_len=1024: int, 
        num_eigh=24: int, 
        auto_reg_k_u=3: int, 
        auto_reg_k_y=2: int, 
        learnable_m_y=True: bool,
    ) -> None:
        super(STU, self).__init__()
        self.d_out = d_out
        self.eigh = stu_utils.get_top_hankel_eigh(input_len, num_eigh)
        self.l, self.k = input_len, num_eigh
        self.auto_reg_k_u = auto_reg_k_u
        self.auto_reg_k_y = auto_reg_k_y
        self.learnable_m_y = learnable_m_y
        self.m_x_var = 1.0 / (float(self.d_out) ** 0.5)

        self.init_m_y = torch.zeros(self.d_out, self.auto_reg_k_y, self.d_out)

        self.init_m_u = stu_utils.get_random_real_matrix(
            (self.d_out, self.d_out, self.auto_reg_k_u), self.m_x_var
        )
        self.init_m_phi = torch.zeros(self.d_out * self.k, self.d_out)

        if learnable_m_y:
            self.m_y = nn.Parameter(self.init_m_y)
        else:
            self.m_y = self.init_m_y

        self.m_u = nn.Parameter(self.init_m_u)
        self.m_phi = nn.Parameter(self.init_m_phi)


    def forward(self, inputs):
        params = (self.m_y, self.m_u, self.m_phi)
        return self.apply_stu(params, inputs, self.eigh)


    @staticmethod
    def apply_stu(params, inputs, eigh):
        """Apply STU.

        Args:
            params: A tuple of parameters of shapes [d_out, d_out], [d_in, d_out, k_u],
            [d_in * k, d_out] and [d_in * k, d_out]
            inputs: Input matrix of shape [l, d_in].
            eigh: A tuple of eigenvalues [k] and circulant eigenvecs [k, l, l].

        Returns:
            A sequence of y_ts of shape [l, d_out].
        """
        m_y, m_u, m_phi = params

        x_tilde = stu_utils.compute_x_tilde(inputs, eigh)

        delta_phi = torch.matmul(x_tilde, m_phi)
        delta_ar_u = stu_utils.compute_ar_x_preds(m_u, inputs)

        return stu_utils.compute_y_t(m_y, delta_phi + delta_ar_u)
