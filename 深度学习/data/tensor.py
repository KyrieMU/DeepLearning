import numpy as np

class Layer():
    def __init__(self):
        self.parameters = list()
    def get_parameters(self):
        return self.parameters
    
class Tensor():
    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children ={}
        if(id is None):
            id = np.random.randint(0, 1000000)
        self.id = id
        
        if(creators is not None):
            for c in creators:
                if (self.id not in c.children):
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
    
    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if (cnt!=0):
                return False
        return True
    
    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="add")
        return Tensor(self.data + other.data)
    

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * (-1),
                          autograd=True,
                          creators=[self],
                          creation_op="neg")
        return Tensor(self.data * (-1))
    
    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="sub")
        return Tensor(self.data - other.data)
    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)
    
    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))
    
    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)
        
        if(self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")
        return Tensor(self.data.transpose())
    
    def mm(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data.dot(other.data),
                          autograd=True,
                          creators=[self, other],
                          creation_op="mm")
        return Tensor(self.data.dot(other.data) )
    

    def sigmoid(self):
        if(self.autograd):
            return Tensor(1.0/(1+np.exp(-self.data)),
                          autograd=True,
                          creators=[self],
                          creation_op="sigmoid")
        return Tensor(1.0/(1+np.exp(-self.data)))

    def tanh(self):
        if(self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op="tanh")
        return Tensor(np.tanh(self.data))

    def relu(self):
        if(self.autograd):
            return Tensor((self.data >= 0)*self.data,
                          autograd=True,
                          creators=[self],
                          creation_op="relu")
        return Tensor((self.data >= 0)*self.data)

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp, axis = len(self.data.shape)-1, keepdims=True)
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p)*target_dist).sum(1).mean()
        
        cnt = 0
        for i in range(len(p)):
            cnt += int(np.argmax(p[i:i+1]) == t[i])

        if(self.autograd):
            out = Tensor(loss, 
                         autograd=True,
                         creators=[self],
                         creation_op="cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            out.cnt = cnt
            return out
        return Tensor(loss)
    

    # ========================================
    # Index_select
    def index_select(self, indices):
        if(self.autograd):
            new = Tensor(self.data[indices.data.flatten()],
                         autograd=True,
                         creators=[self],
                         creation_op="index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data.flatten()])
    # ===========================================
    
    def softmax(self):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape)-1,
                                       keepdims=True)
        return softmax_output    
    
    def backward(self, grad=None, grad_origin=None):
        if(self.autograd):
            if(grad_origin is not None):
                if(self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1
                    
            if(grad is None):
                grad = Tensor(np.ones_like(self.data))   
                
            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad
            
            if(self.creators is not None and \
                  (self.all_children_grads_accounted_for() or grad_origin is None)):
            
                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)


                if(self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__(), self)

                if(self.creation_op == "sub"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)

                if(self.creation_op == "mul"):
                    new = Tensor(self.grad.data * self.creators[1].data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.data * self.creators[0].data)
                    self.creators[1].backward(new, self)

                if(self.creation_op == "mm"):
                    layer = self.creators[0].data
                    weights = self.creators[1].data
                    new = Tensor(self.grad.data.dot(weights.transpose()))
                    self.creators[0].backward(new, self)
                    new = Tensor(layer.transpose().dot(self.grad.data))
                    self.creators[1].backward(new, self)

                if(self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose(), self)

                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    copies = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim, copies), self)

                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim), self)


                if(self.creation_op == "sigmoid"):
                    ones = np.ones_like(self.grad.data)
                    new = Tensor(self.grad.data * (self.data * (ones - self.data)))
                    self.creators[0].backward(new, self)

                if(self.creation_op == "tanh"):
                    ones = np.ones_like(self.grad.data)
                    new = Tensor(self.grad.data * (ones - self.data*self.data))
                    self.creators[0].backward(new, self)


                if(self.creation_op == "relu"):
                    new = Tensor(self.grad.data * (self.data >= 0))
                    self.creators[0].backward(new, self)

                if(self.creation_op == "cross_entropy"):
                    dx = self.softmax_output - self.target_dist
                    self.creators[0].backward(Tensor(dx), self)
                
                # =============================================
                # index select的导数
                if(self.creation_op == "index_select"):
                    new_grad = np.zeros_like(self.creators[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.creators[0].backward(Tensor(new_grad), self)
                # ==============================================
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
  



    
class Linear(Layer):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        
        self.use_bias = bias
        
        w = np.random.randn(n_inputs, n_outputs)*np.sqrt(2.0/n_inputs)
        self.weights = Tensor(w, autograd=True)
        self.parameters.append(self.weights)
        
        if(self.use_bias):
            self.bias = Tensor(np.zeros(n_outputs), autograd=True)
            self.parameters.append(self.bias)
        
    def forward(self, input):
        if(self.use_bias):
            return input.mm(self.weights) + self.bias.expand(0, len(input.data))
        else:
            return input.mm(self.weights)
        
class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        self.layers = layers
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params
    
class Tanh(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input.tanh()
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input.sigmoid()  
    
class Relu(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input.relu()   
    
    
class CrossEntropyLoss(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):
        return input.cross_entropy(target)   
    
class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
    
        self.vocab_size = vocab_size
        self.dim = dim
        
        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)
        self.parameters.append(self.weight)
    
    def forward(self, input): 
        return self.weight.index_select(input)
    
class RNNCell(Layer):
    def __init__(self, n_inputs, n_hidden, n_output, activation="tanh"):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        if(activation == "sigmoid"):
            self.activation = Sigmoid()
        elif(activation == "tanh"):
            self.activation = Tanh()
        elif(activation == "relu"):
            self.activation = Relu()
        else:
            raise Exception("Non-Linearity not found")
            
        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)
        
        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()
        
    def forward(self, input, hidden, out=True):
        combined = self.w_ih.forward(input) + self.w_hh.forward(hidden)
        new_hidden = self.activation.forward(combined)

        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
    
class SGD():
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr
        
    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.lr
            if(zero):
                p.grad.data *= 0


