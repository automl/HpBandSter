--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found here
--  https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Code modified for Shake-Shake by Xavier Gastaldi
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels
   local k = opt.baseWidth
   local forwardShake = opt.forwardShake
   local backwardShake = opt.backwardShake
   local shakeImage = opt.shakeImage
   local batchSize = opt.batchSize

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      local useConv = shortcutType == 'C' or
         (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
      if useConv then
         -- 1x1 convolution
         return nn.Sequential()
            :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
            :add(SBatchNorm(nOutputPlane))
      elseif nInputPlane ~= nOutputPlane then
         -- Strided, zero-padded identity shortcut
         return nn.Sequential()
            :add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
               :add(nn.Identity())
               :add(nn.MulConstant(0)))
      else
         return nn.Identity()
      end
   end

   -- The aggregated residual transformation bottleneck layer, Form (C)
   local function resnext_bottleneck_C(n, stride, forwardShake, backwardShake, shakeImage, batchSize)
      local nInputPlane = iChannels
      iChannels = n * 4

      local D = math.floor(n * (opt.baseWidth/64))
      local C = opt.groups

      local s1 = nn.Sequential()
      s1:add(Convolution(nInputPlane,D*C,1,1,1,1,0,0))
      s1:add(SBatchNorm(D*C))
      s1:add(ReLU(true))
      s1:add(Convolution(D*C,D*C,3,3,stride,stride,1,1,C))
      s1:add(SBatchNorm(D*C))
      s1:add(ReLU(true))
      s1:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s1:add(SBatchNorm(n*4))

      local s2 = nn.Sequential()
      s2:add(Convolution(nInputPlane,D*C,1,1,1,1,0,0))
      s2:add(SBatchNorm(D*C))
      s2:add(ReLU(true))
      s2:add(Convolution(D*C,D*C,3,3,stride,stride,1,1,C))
      s2:add(SBatchNorm(D*C))
      s2:add(ReLU(true))
      s2:add(Convolution(D*C,n*4,1,1,1,1,0,0))
      s2:add(SBatchNorm(n*4))

      return nn.Sequential()
        :add(nn.ConcatTable()
            :add(shortcut(nInputPlane, n * 4, stride))
            :add(s1)
            :add(s2))
         :add(nn.ShakeShakeTable(0.5, forwardShake, backwardShake, shakeImage, batchSize))
         :add(nn.CAddTable(false))
         :add(ReLU(true))
   end


   local function layerC10(block, nInputPlane, nOutputPlane, count, stride, forwardShake, backwardShake, shakeImage, batchSize)
      local s = nn.Sequential()

      if count < 1 then
        return s
      end

      s:add(block(nInputPlane, nOutputPlane, stride, forwardShake, backwardShake, shakeImage, batchSize))

      for i=2,count do
        s:add(block(nOutputPlane, nOutputPlane, 1, forwardShake, backwardShake, shakeImage, batchSize))
      end

      return s
   end

   -- Creates count residual blocks with specified number of features
   local function layerC100(block, features, count, stride, forwardShake, backwardShake, shakeImage, batchSize)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, i == 1 and stride or 1, forwardShake, backwardShake, shakeImage, batchSize))
      end
      return s
   end

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      error('ImageNet is not yet implemented with Shake-Shake')
   elseif opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
      local n = (depth - 2) / 6
      iChannels = 16
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(Convolution(3,16,3,3,1,1,1,1))
      model:add(ShareGradInput(SBatchNorm(16), 'first'))
      model:add(layerC10(nn.ShakeShakeBlock, 16, k, n, 1, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layerC10(nn.ShakeShakeBlock, k, 2*k, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layerC10(nn.ShakeShakeBlock, 2*k, 4*k, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(4*k):setNumInputDims(3))
      model:add(nn.Linear(4*k, 10))

   elseif opt.dataset == 'cifar100' then
      bottleneck = resnext_bottleneck_C
      assert((depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101')
      local n = (depth - 2) / 9
      iChannels = k
      print(' | ResNet-' .. depth .. ' ' .. opt.dataset)

      model:add(Convolution(3,64,3,3,1,1,1,1))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(layerC100(bottleneck, 64, n, 1, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layerC100(bottleneck, 128, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(layerC100(bottleneck, 256, n, 2, forwardShake, backwardShake, shakeImage, batchSize))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(1024):setNumInputDims(3))
      model:add(nn.Linear(1024, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:type(opt.tensorType)

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel
