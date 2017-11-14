---
--- Created by pwang8.
--- DateTime: 11/8/17 4:07 PM
--- This file is created for testing the torch_rhn_ptb.lua script.
---

--local ok,cunn = pcall(require, 'fbcunn')
--if not ok then
--    ok,cunn = pcall(require,'cunn')
--    if ok then
--        print("warning: fbcunn not found. Falling back to cunn")
--        LookupTable = nn.LookupTable
--    else
--        print("Could not find cunn or fbcunn. Either is required")
--        os.exit()
--    end
--else
--    deviceParams = cutorch.getDeviceProperties(1)
--    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
--    LookupTable = nn.LookupTable
--end
require('nn')
LookupTable = nn.LookupTable    -- They previously use and only use cunn or fbcunn. My laptop does not have cuda supported, so get rid of it right now
                                -- right now it's just checking out the model on cpu mode
require('nngraph')
require('base')
local ptb = require('data')

-- Attention: pwang8. The recurrence_depth is a different param from the layers param.
-- The recurrence_depth param is the recurrent depth in each time step. The layers param
-- is the vertical layers of stacked RHNs as shown in the readme Deep Transition RNN structure
local params = {batch_size=20,
    seq_length=35,
    layers=1,
    decay=1.02,
    rnn_size=1000,
    dropout_x=0.25,
    dropout_i=0.75,
    dropout_h=0.25,
    dropout_o=0.75,
    init_weight=0.04,
    lr=0.2,
    vocab_size=10000,
    max_epoch=20,
    max_max_epoch=1000,
    max_grad_norm=10,
    weight_decay=1e-7,
    recurrence_depth=10,
    initial_bias=-4}


local disable_dropout = false
local function local_Dropout(input, noise)
    return nn.CMulTable()({input, noise})
end

local function transfer_data(x)
    --return x:cuda()
    return x    -- cuda not supported on laptop, not set any gpu module right now
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

-- The param prev_c is the value of the c-gate in rhn (Equation 9). Interestingly it is never directly used in the script.
-- It is only sent into the function.
local function rhn(x, prev_c, prev_h, noise_i, noise_h)
    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slice the n_gates dimension, i.e dimension 2
    local reshaped_noise_i = nn.Reshape(2,params.rnn_size)(noise_i)   -- this might mean rhn has 2 gates, and this is the noise mask for input
    local reshaped_noise_h = nn.Reshape(2,params.rnn_size)(noise_h)   -- this should be the noise for prior hidden state
    local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i)   -- SplitTable(2) means split the input tensor along the 2nd dim, which is the num of gates dim
    local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)   -- after SplitTable, the output is a table of tensors
    -- Calculate all two gates
    local dropped_h_tab = {}
    local h2h_tab = {}
    local t_gate_tab = {}
    local c_gate_tab = {}
    local in_transform_tab = {}
    local s_tab = {}
    for layer_i = 1, params.recurrence_depth do -- Attention: pwang8. The for loop iteration count is the recurrence_depth (horizontal, in each time step), not RHN layers
        local i2h        = {}   -- i2h is a single item, bcz only one rhn unit is adopted to connect vertical layers (vertical input of x)
        h2h_tab[layer_i] = {}   -- h2h is of multiple items, bcz a large (10 in this example) recurrent depth is adopted (strictly it is not the (vertical) layers of NN, it is the horizontal depth defined in transition RNN in between one time step)
        if layer_i == 1 then
            for i = 1, 2 do
                -- Use select table to fetch each gate
                local dropped_x         = local_Dropout(x, nn.SelectTable(i)(sliced_noise_i)) -- slidced_noise_i is a table of tensors. So there are 2 gates and corresponding noise mask
                dropped_h_tab[layer_i]  = local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))  -- the 2 gates contain one gate for calc hidden state, and the other gate being the transform gate
                i2h[i]                  = nn.Linear(params.rnn_size, params.rnn_size)(dropped_x)    -- there are two i2h and h2h_tab bcz in equation 7 and 8 x and hidden state_h are utilized twice (2 sets of matrix multiplication)
                h2h_tab[layer_i][i]     = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h_tab[layer_i])
            end
            t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(nn.CAddTable()({i2h[1], h2h_tab[layer_i][1]}))) -- this is the tranform module in equation 8 in the paper. I guess the AddConstant is an init step
            in_transform_tab[layer_i] = nn.Tanh()(nn.CAddTable()({i2h[2], h2h_tab[layer_i][2]}))  -- calculate the hidden module, depicted in equation 7 in the paper
            c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i])) -- in the implementation, the c gate is designed as (1-t), in which the t gate is calculated aboved
            s_tab[layer_i]           = nn.CAddTable()({
                nn.CMulTable()({c_gate_tab[layer_i], prev_h}),
                nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
            })  -- calc the output at time step t, as depicted in equation 6 in the paper
        else
            for i = 1, 2 do
                -- Use select table to fetch each gate
                dropped_h_tab[layer_i]  = local_Dropout(s_tab[layer_i-1], nn.SelectTable(i)(sliced_noise_h))
                h2h_tab[layer_i][i]     = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h_tab[layer_i]) -- h2h_tab[layer_i][1] is the multiplication in equation 8, h2h_tab[layer_i][2] is multiplication in equation 7
            end
            t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(h2h_tab[layer_i][1]))   -- Attention: refer to the Deep Transition RNN figure in readme file to check the structure here    -- Equation 8
            in_transform_tab[layer_i] = nn.Tanh()(h2h_tab[layer_i][2])  -- for transition layers inside one time step, only the h2h state values (horizontal) are propagated. So, it's a little different from the first transition layer   -- Equation 7
            c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i]))     -- Equation 9, with the simplified assumption that c = 1 - t
            s_tab[layer_i]           = nn.CAddTable()({
                nn.CMulTable()({c_gate_tab[layer_i], s_tab[layer_i-1]}),
                nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
            })  -- Equation 6
        end
    end
    local next_h = s_tab[params.recurrence_depth]
    local next_c = prev_c   -- this is a little weird. prev_c seems not used at all. It's also weird why the author liked to keep the record of the c gate
    return next_c, next_h
end

local function create_network()
    local x                = nn.Identity()()    -- input of rhn_network
    local y                = nn.Identity()()    -- output of rhn_network, may not be the output of this whole NN created from this function
    local prev_s           = nn.Identity()()    -- previous hidden state s from each rhn (vertical) layer. the prev_s contains two parts. One is the c-gate value, the other is the hidden s_state
    local noise_x          = nn.Identity()()    -- the following 4 are dropout masks. This is the dropout mask for (after) the input layer into the rhn module
    local noise_i          = nn.Identity()()    -- the dropout mask (before) entering the hidden layer. It doubles the size of rnn_size, bcz we use this input twice to calculate hidden state_s in rhn module and the t_gate. I don't quite understand if it's necessary to have both noise_i mask and noise_x mask
    local noise_h          = nn.Identity()()    -- the dropout mask for (before) the hidden layer. It doubles the size of rnn_size, bcz we use this hidden state_h twice to calculate hidden state_s in rhn module and the t_gate.
    local noise_o          = nn.Identity()()    -- dropout mask for the output of rhn (it's the state_s of the highest layer in rhn_network)
    local i                = {[0] = LookupTable(params.vocab_size,
                                params.rnn_size)(x)}    -- this lookup table is specifically designed for language model
    i[0] = local_Dropout(i[0], noise_x)
    local next_s           = {} -- the stored state_s contains two parts for each hidden layer. 1st part is the c_gate value, the 2nd part is the hidden state_s value from rhn
    local split            = {prev_s:split(2 * params.layers)}  -- the split function is the split() for nngraph.Node. Can be found here: https://github.com/torch/nngraph/blob/master/node.lua (This is not the split function for tensor)
    local noise_i_split    = {noise_i:split(params.layers)} -- this nngraph.Node.split() function returns noutput number of new nodes that each take a single component of the output of this
    local noise_h_split    = {noise_h:split(params.layers)} -- node in the order they are returned.
    for layer_idx = 1, params.layers do     -- this params.layers is the vertical layer number of rhn, not the recurrent depth.
        local prev_c         = split[2 * layer_idx - 1]     -- the prev_c is the c_gate value from previous time step (I don't think it is actually used in rhn)
        local prev_h         = split[2 * layer_idx]         -- the prev_h is the hidden state_s value from previous time step. Here it does not concern recurrent depth, which is sth studied inside rhn
        local n_i            = noise_i_split[layer_idx]     -- n_i and n_h are the dropout mask. n_i is the dropout mask for each (vertical) rhn layer's (vertical) input
        local n_h            = noise_h_split[layer_idx]     -- n_h is the dropout mask for each (horizontal) rhn unit. I'm not sure if dropout is adopted inside one time step of rnn (in recurrence_depth)
        local next_c, next_h = rhn(i[layer_idx - 1], prev_c, prev_h, n_i, n_h)
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h   -- this next_h is the state_s value, which is the output of rhn module
    end
    local dropped          = local_Dropout(i[params.layers], noise_o)   -- the output of rhn module, after dropout
    local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
    local pred             = nn.LogSoftMax()(h2y(dropped))  -- change the dimension from rnn_size to vocab_size and add LogSoftMax
    local err              = nn.ClassNLLCriterion()({pred, y})  -- the structure of this nn is a little different. It has labels (y) also as input, and the output of the whole nn is a table {error, next_hidden_s}
    local module           = nn.gModule({x, y, prev_s, noise_x, noise_i, noise_h, noise_o},
                                        {err, nn.Identity()(next_s)})
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

local function setup()
    print("Creating an RHN network.")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}    -- So this s table records all state_s values for all time step
    model.ds = {}   -- I guess this is derivatives over hidden states (s).
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end

    model.noise_i = {}
    model.noise_x = {}
    model.noise_xe = {}
    for j = 1, params.seq_length do
        model.noise_x[j] = transfer_data(torch.zeros(params.batch_size, 1))
        model.noise_xe[j] = torch.expand(model.noise_x[j], params.batch_size, params.rnn_size)  -- the expand() duplicate the original tensor, without allocating new memory
        model.noise_xe[j] = transfer_data(model.noise_xe[j])
    end
    model.noise_h = {}
    for d = 1, params.layers do
        model.noise_i[d] = transfer_data(torch.zeros(params.batch_size, 2 * params.rnn_size))
        model.noise_h[d] = transfer_data(torch.zeros(params.batch_size, 2 * params.rnn_size))
    end
    model.noise_o = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))

    model.pred = {}
    for j = 1, params.seq_length do
        model.pred[j] = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
    end
    local y                = nn.Identity()()
    local pred             = nn.Identity()()
    local err              = nn.ClassNLLCriterion()({pred, y})
    model.test             = transfer_data(nn.gModule({y, pred}, {err}))
end

local function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

local function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

-- convenience functions to handle noise
local function sample_noise(state)
    for i = 1, params.seq_length do
        model.noise_x[i]:bernoulli(1 - params.dropout_x)
        model.noise_x[i]:div(1 - params.dropout_x)
    end
    -- The noise_x setting is a little confusing. I don't quite understand it. It looks like in the sequence they randomly turn off certain words in a word sequence
    for b = 1, params.batch_size do
        for i = 1, params.seq_length do
            local x = state.data[state.pos + i - 1]
            for j = i+1, params.seq_length do
                if state.data[state.pos + j - 1] == x then
                    model.noise_x[j][b] = model.noise_x[i][b]
                    -- we only need to override the first time; afterwards subsequent are copied:
                    break
                end
            end
        end
    end
    for d = 1, params.layers do
        model.noise_i[d]:bernoulli(1 - params.dropout_i)
        model.noise_i[d]:div(1 - params.dropout_i)
        model.noise_h[d]:bernoulli(1 - params.dropout_h)
        model.noise_h[d]:div(1 - params.dropout_h)
    end
    model.noise_o:bernoulli(1 - params.dropout_o)
    model.noise_o:div(1 - params.dropout_o)
end

local function reset_noise()
    for j = 1, params.seq_length do
        model.noise_x[j]:zero():add(1)
    end
    for d = 1, params.layers do
        model.noise_i[d]:zero():add(1)
        model.noise_h[d]:zero():add(1)
    end
    model.noise_o:zero():add(1)
end

local function fp(state)
    g_replace_table(model.s[0], model.start_s)
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end

    if disable_dropout then reset_noise() else sample_noise(state) end
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i] = unpack(model.rnns[i]:forward(
        {x, y, s, model.noise_xe[i], model.noise_i, model.noise_h, model.noise_o}))
        state.pos = state.pos + 1
    end
    g_replace_table(model.start_s, model.s[params.seq_length])
    return model.err
end

local function bp(state)
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        local derr = transfer_data(torch.ones(1))
        local tmp = model.rnns[i]:backward( -- Yarin: do we need model.noise_x[i+1]?
                            {x, y, s, model.noise_xe[i], model.noise_i, model.noise_h, model.noise_o},
                            {derr, model.ds})[3]
        g_replace_table(model.ds, tmp)
        cutorch.synchronize()
    end
    state.pos = state.pos + params.seq_length
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    paramx:add(paramdx:mul(-params.lr))   -- They only use SGD and param weigth decay, interesting
    paramx:add(-params.weight_decay, paramx)    -- the weight decay is interesting. This is different from L2 norm. Some intro can be found here https://metacademy.org/graphs/concepts/weight_decay_neural_networks
end

local function run_valid()
    reset_state(state_valid)
    disable_dropout = true  -- Attention: in validation, dropout is diabled, which means all inputs go through. This is the so-called variational dropout
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        local p = fp(state_valid)
        perp = perp + p:mean()
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    disable_dropout = false
end

local function run_test()
    reset_state(state_test)
    reset_noise()   -- this is the same as disable dropout
    local perp = 0
    local len = state_test.data:size(1)
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward(
                                    {x, y, model.s[0], model.noise_xe[1], model.noise_i, model.noise_h, model.noise_o}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
end

local function main()
    --g_init_gpu(1)
    state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
    state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
    state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}
    print("Network parameters:")
    print(params)
    local states = {state_train, state_valid, state_test}
    for _, state in pairs(states) do
        reset_state(state)
    end
    setup()
    local step = 0
    local epoch = 0
    local total_cases = 0
    local beginning_time = torch.tic()
    local start_time = torch.tic()
    print("Starting training.")
    local epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
    local perps
    while epoch < params.max_max_epoch do
        local perp = fp(state_train):mean()
        if perps == nil then
            perps = torch.zeros(epoch_size):add(perp)
        end
        perps[step % epoch_size + 1] = perp
        step = step + 1
        bp(state_train)
        total_cases = total_cases + params.seq_length * params.batch_size
        epoch = step / epoch_size
        if step % torch.round(epoch_size / 10) == 10 then
            local wps = torch.floor(total_cases / torch.toc(start_time))
            local since_beginning = g_d(torch.toc(beginning_time) / 60)
            print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
        end
        if step % epoch_size == 0 then
            run_valid()
            run_test()
            if epoch > params.max_epoch then
                params.lr = params.lr / params.decay
            end
        end
        if step % 33 == 0 then
            cutorch.synchronize()
            collectgarbage()
        end
    end
    run_test()
    print("Training is over.")
end

main()
