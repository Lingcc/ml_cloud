import torch 
import torch._dynamo as dynamo
#print(torch.modules)
print(torch.__version__)
dynamo.verbose=True

def my_compiler(gm: torch.fx.GraphModule,
                example_inputs: list[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    scripted = torch.jit.trace(gm, example_inputs)
    return torch.jit.optimize_for_inference(scripted)
    #return gm.forward # returns a python callable


@torch.compile(backend=my_compiler)
def toy_example(a, b):
    x = a/(torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * 1
    return x*b


print(toy_example(torch.randn(4), torch.randn(4)))
