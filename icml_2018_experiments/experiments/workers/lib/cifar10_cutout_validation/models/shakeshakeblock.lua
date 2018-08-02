--
--  Copyright (c) 2017, Xavier Gastaldi.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'nn'
require 'cudnn'
require 'cunn'
require 'models/mulconstantslices'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local ShakeShakeBlock, parent = torch.class('nn.ShakeShakeBlock', 'nn.Container')

function ShakeShakeBlock:__init(nInputPlane, nOutputPlane, stride, forwardShake, backwardShake, shakeImage, batchSize)

   parent.__init(self)
   self.gradInput = torch.Tensor()
   self.train = true
   self.forwardShake = forwardShake
   self.backwardShake = backwardShake
   self.shakeImage = shakeImage
   self.batchSize = batchSize

   -- Residual branch #1
   self.net1 = nn.Sequential()
   self.net1:add(ReLU(true))
   self.net1:add(Convolution(nInputPlane, nOutputPlane,3,3,stride,stride,1,1))
   self.net1:add(SBatchNorm(nOutputPlane))
   self.net1:add(ReLU(true))
   self.net1:add(Convolution(nOutputPlane, nOutputPlane,3,3,1,1,1,1))
   self.net1:add(SBatchNorm(nOutputPlane))
   if self.shakeImage == 'true' then -- If we want to work at the image level
     self.net1:add(nn.MulConstantSlices(torch.ones(self.batchSize)))
   else -- If we want to work at the batch level
     self.net1:add(nn.MulConstant(0.5))
   end
    
   -- Residual branch #2
   self.net2 = nn.Sequential()
   self.net2:add(ReLU(true))
   self.net2:add(Convolution(nInputPlane, nOutputPlane,3,3,stride,stride,1,1))
   self.net2:add(SBatchNorm(nOutputPlane))
   self.net2:add(ReLU(true))
   self.net2:add(Convolution(nOutputPlane, nOutputPlane,3,3,1,1,1,1))
   self.net2:add(SBatchNorm(nOutputPlane))
   if self.shakeImage == 'true' then -- If we want to work at the image level
     self.net2:add(nn.MulConstantSlices(torch.ones(self.batchSize)))
   else -- If we want to work at the batch level
     self.net2:add(nn.MulConstant(0.5))
   end

   -- Skip connection
   self.skip = nn.Sequential()

   if nInputPlane == nOutputPlane then
     self.skip:add(nn.Identity())
   else
     self.skip:add(ReLU(true))

     -- Skip path #1
     s1 = nn.Sequential()
     s1:add(nn.SpatialAveragePooling(1, 1, stride, stride))
     s1:add(Convolution(nInputPlane, nOutputPlane/2, 1,1, 1,1, 0,0))

     -- Skip path #2
     s2 = nn.Sequential()
     -- Shift the tensor by one pixel right and one pixel down (to make the 2nd path "see" different pixels)
     s2:add(nn.SpatialZeroPadding(1, -1, 1, -1))
     s2:add(nn.SpatialAveragePooling(1, 1, stride, stride))
     s2:add(Convolution(nInputPlane, nOutputPlane/2, 1,1, 1,1, 0,0))

     -- Concatenate the 2 paths along the width dimension
     self.skip:add(nn.ConcatTable()
          :add(s1)
          :add(s2))
     :add(nn.JoinTable(2))

     self.skip:add(SBatchNorm(nOutputPlane))
   end
    
   self.modules = {self.net1, self.net2, self.skip}

end

function ShakeShakeBlock:updateOutput(input)

   local skip_forward = self.skip:forward(input)
   self.output:resizeAs(skip_forward):copy(skip_forward)
   if self.shakeImage == 'true' then -- If we want to work at the image level
     if self.train and self.forwardShake == 'true' then -- If we are training and want to randomize the forward pass
       self.alpha = torch.rand(self.batchSize) -- then create a vector of random numbers
       -- Access the vector where constants are stored in self.net1 nn.MulConstantSlices and overwrite it with self.alpha
       self.net1.modules[7].constant_tensor = self.alpha  
       -- Access the vector where constants are stored in self.net2 nn.MulConstantSlices and overwrite it with 1 - self.alpha
       self.net2.modules[7].constant_tensor = torch.ones(self.batchSize) - self.alpha
     else -- If we are testing or do not want to randomize the forward pass
       -- Access the vector where constants are stored in in self.net1 nn.MulConstantSlices and overwrite it with 0.5s
       self.net1.modules[7].constant_tensor = torch.ones(self.batchSize)*0.5
       -- Access the vector where constants are stored in self.net2 nn.MulConstantSlices and overwrite it with 0.5s
       self.net2.modules[7].constant_tensor = torch.ones(self.batchSize)*0.5
     end
   else -- If we want to work at the batch level
     if self.train and self.forwardShake == 'true' then -- If we are training and want to randomize the forward pass
       self.alpha = torch.rand(1)[1] -- then create a random number
       -- Access the constant in self.net1 nn.MulConstant and overwrite it with self.alpha
       self.net1.modules[7].constant_scalar = self.alpha
       --print('batch forward')
       -- Access the constant in self.net2 nn.MulConstant and overwrite it with 1 - self.alpha
       self.net2.modules[7].constant_scalar = 1 - self.alpha
     else -- If we are testing or do not want to randomize the forward pass
       -- Access the constant in self.net1 nn.MulConstant and overwrite it with 0.5
       self.net1.modules[7].constant_scalar = 0.5
       -- Access the constant in self.net2 nn.MulConstant and overwrite it with 0.5
       self.net2.modules[7].constant_scalar = 0.5
     end
   end

   -- Now that the constants have been updated, forward self.net1 and self.net2 and add the results to self.output
   self.output:add(self.net1:forward(input))
   self.output:add(self.net2:forward(input))

   return self.output
end

function ShakeShakeBlock:updateGradInput(input, gradOutput)

   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input):copy(self.skip:updateGradInput(input, gradOutput))

   if self.shakeImage == 'true' then -- If we want to work at the image level
     if self.backwardShake == 'true' then -- If we want to randomize the backward pass
       self.beta = torch.rand(self.batchSize) -- then create a vector of random numbers
       -- Access the vector where constants are stored in self.net1 nn.MulConstantSlices and overwrite it with self.beta
       self.net1.modules[7].constant_tensor = self.beta
       -- Access the vector where constants are stored in self.net2 nn.MulConstantSlices and overwrite it with 1 - self.beta
       self.net2.modules[7].constant_tensor = torch.ones(self.batchSize) - self.beta
     else -- If we do not want to randomize the backward pass
       -- Access the vector where constants are stored in in self.net1 nn.MulConstantSlices and overwrite it with 0.5s
       self.net1.modules[7].constant_tensor = torch.ones(self.batchSize)*0.5
       -- Access the vector where constants are stored in in self.net2 nn.MulConstantSlices and overwrite it with 0.5s
       self.net2.modules[7].constant_tensor = torch.ones(self.batchSize)*0.5
     end
   else -- If we want to work at the batch level
     if self.backwardShake == 'true' then -- If we want to randomize the backward pass
       self.beta = torch.rand(1)[1] -- then create a random number
       -- Access the constant in self.net1 nn.MulConstant and overwrite it with self.beta
       --print('batch backward')
       self.net1.modules[7].constant_scalar = self.beta
       -- Access the constant in self.net2 nn.MulConstant and overwrite it with 1 - self.beta
       self.net2.modules[7].constant_scalar = 1 - self.beta
     else -- If we do not want to randomize the backward pass
       -- Access the constant in self.net1 nn.MulConstant and overwrite it with 0.5
       self.net1.modules[7].constant_scalar = 0.5
       -- Access the constant in self.net2 nn.MulConstant and overwrite it with 0.5
       self.net2.modules[7].constant_scalar = 0.5
     end
   end

   -- Now that the constants have been updated, "backward" self.net1 and self.net2 and add the results to self.gradInput
   self.gradInput:add(self.net1:updateGradInput(input, gradOutput))
   self.gradInput:add(self.net2:updateGradInput(input, gradOutput))

   return self.gradInput
end

function ShakeShakeBlock:accGradParameters(input, gradOutput, scale)

   scale = scale or 1
   self.skip:accGradParameters(input, gradOutput, scale)
   self.net1:accGradParameters(input, gradOutput, scale)
   self.net2:accGradParameters(input, gradOutput, scale)

end

