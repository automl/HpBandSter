--
--  Copyright (c) 2017, Xavier Gastaldi.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local ShakeShakeTable, parent = torch.class('nn.ShakeShakeTable', 'nn.Module')

function ShakeShakeTable:__init(constant_scalar, forwardShake, backwardShake, shakeImage, batchSize)

   parent.__init(self)
   assert(type(constant_scalar) == 'number', 'input is not scalar!')

   self.forwardShake = forwardShake
   self.backwardShake = backwardShake
   self.shakeImage = shakeImage
   self.batchSize = batchSize
   self.constant_tensor = torch.ones(batchSize)

   self.output = {}
   self.output[1] = torch.Tensor():cuda()
   self.output[2] = torch.Tensor():cuda()
   self.output[3] = torch.Tensor():cuda()
   self.gradInput = {}
   self.gradInput[1] = torch.Tensor():cuda()
   self.gradInput[2] = torch.Tensor():cuda()
   self.gradInput[3] = torch.Tensor():cuda()

end

function ShakeShakeTable:updateOutput(input)

   if self.shakeImage == 'true' then -- If we want to work at the image level
     if self.train and self.forwardShake == 'true' then -- If we are training and want to randomize the forward pass
       self.constant_tensor = torch.rand(self.batchSize)
     else -- If we are testing or do not want to randomize the forward pass
       self.constant_tensor = torch.ones(self.batchSize)*0.5
     end
   else -- If we want to work at the batch level
     if self.train and self.forwardShake == 'true' then -- If we are training and want to randomize the forward pass
       self.alpha = torch.rand(1)[1] -- then create a random number
       self.constant_tensor = torch.ones(self.batchSize)*self.alpha
     else -- If we are testing or do not want to randomize the forward pass
       self.constant_tensor = torch.ones(self.batchSize)*0.5
     end
   end

    self.output[1]:resizeAs(input[1])
    self.output[1]:copy(input[1])

    self.output[2]:resizeAs(input[2])
    self.output[2]:copy(input[2])
    mul_slices(self.output[2], self.constant_tensor)

    self.output[3]:resizeAs(input[3])
    self.output[3]:copy(input[3])
    mul_slices(self.output[3], torch.ones(self.batchSize)-self.constant_tensor)

  return self.output
end

function ShakeShakeTable:updateGradInput(input, gradOutput)
  if self.gradInput then

      if self.shakeImage == 'true' then -- If we want to work at the image level
        if self.backwardShake == 'true' then -- If we want to randomize the backward pass
          self.constant_tensor = torch.rand(self.batchSize)
        else -- If we do not want to randomize the backward pass
          self.constant_tensor = torch.ones(self.batchSize)*0.5
        end
      else -- If we want to work at the batch level
        if self.backwardShake == 'true' then -- If we want to randomize the backward pass
          self.beta = torch.rand(1)[1] -- then create a random number
          self.constant_tensor = torch.ones(self.batchSize)*self.beta
        else -- If we do not want to randomize the backward pass
          self.constant_tensor = torch.ones(self.batchSize)*0.5
        end
      end

      self.gradInput[1]:resizeAs(gradOutput[1])
      self.gradInput[1]:copy(gradOutput[1])

      self.gradInput[2]:resizeAs(gradOutput[2])
      self.gradInput[2]:copy(gradOutput[2])
      mul_slices(self.gradInput[2], self.constant_tensor)

      self.gradInput[3]:resizeAs(gradOutput[3])
      self.gradInput[3]:copy(gradOutput[3])
      mul_slices(self.gradInput[3], torch.ones(self.batchSize)-self.constant_tensor)

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
