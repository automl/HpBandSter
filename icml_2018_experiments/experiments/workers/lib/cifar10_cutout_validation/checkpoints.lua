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

local checkpoint = {}

------Shake-Shake------
require 'models/shakeshakeblock'
require 'models/shakeshaketable'
local std = require 'std'
------Shake-Shake------

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end

   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

function checkpoint.save(epoch, model, optimState, isLastModel, opt)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   ------Shake-Shake------
   --model = deepCopy(model):float():clearState()
   model = std.tree.clone(model):float():clearState()
   ------Shake-Shake------

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   ------Shake-Shake------
   --torch.save(paths.concat(opt.save, modelFile), model)
   --torch.save(paths.concat(opt.save, optimFile), optimState)
   --torch.save(paths.concat(opt.save, 'latest.t7'), {
   --   epoch = epoch,
   --   modelFile = modelFile,
   --   optimFile = optimFile,
   --})
   ------Shake-Shake------

   if isLastModel then
      --torch.save(paths.concat(opt.save, 'model_last.t7'), model)
      ------Shake-Shake------
      --torch.save(paths.concat(opt.save, 'optimState_last.t7'), optimState)
      ------Shake-Shake------
      torch.save(paths.concat(opt.save, modelFile), model)
      torch.save(paths.concat(opt.save, optimFile), optimState)
   end
end

return checkpoint
