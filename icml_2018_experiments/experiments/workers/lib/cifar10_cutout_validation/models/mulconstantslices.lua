--
--  Copyright (c) 2017, Xavier Gastaldi.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
-- Similar to MulConstant but multiplies each image in the mini-batch by a different constant
-- Constants are stored in self.constant_tensor
--

local MulConstantSlices, parent = torch.class('nn.MulConstantSlices', 'nn.Module')

function MulConstantSlices:__init(constant_tensor,ip)
  parent.__init(self)

  self.constant_tensor = constant_tensor

  -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end

end

function MulConstantSlices:updateOutput(input)
  if self.inplace then
    mul_slices(input, self.constant_tensor)
    self.output:set(input)
  else
    self.output:resizeAs(input)
    self.output:copy(input)
    mul_slices(self.output, self.constant_tensor)
  end
  return self.output
end

function MulConstantSlices:updateGradInput(input, gradOutput)
  if self.gradInput then
    if self.inplace then
      mul_slices(gradOutput, self.constant_tensor)
      self.gradInput:set(gradOutput)
      div_slices(input, self.constant_tensor)
    else
      self.gradInput:resizeAs(gradOutput)
      self.gradInput:copy(gradOutput)
      mul_slices(self.gradInput, self.constant_tensor)
    end
    return self.gradInput
  end
end

function mul_slices(tensor, vec)
    for i, slice in ipairs(tensor:split(1, 1)) do
        slice:mul(vec[i])
    end
    return tensor
end

function div_slices(tensor, vec)
    for i, slice in ipairs(tensor:split(1, 1)) do
        slice:div(vec[i])
    end
    return tensor
end
