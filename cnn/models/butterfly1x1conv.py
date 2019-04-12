from butterfly import Butterfly


class Butterfly1x1Conv(Butterfly):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    """

    def forward(self, input):
        """
        Parameters:
            input: (batch, c, h, w) if real or (batch, c, h, w, 2) if complex
        Return:
            output: (batch, nstack * c, h, w) if real or (batch, nstack * c, h, w, 2) if complex
        """
        # TODO: Only doing real for now
        batch, c, h, w = input.shape
        input_reshape = input.view(batch, c, h * w).transpose(1, 2).reshape(-1, c)
        output = super().forward(input_reshape)
        return output.view(batch, h * w, self.nstack * c).transpose(1, 2).view(batch, self.nstack * c, h, w)
