import numpy as np
import torch


class HookModule:
    def __init__(self, model, module):
        self.model = model
        self.activations = None

        module.register_forward_hook(self._hook_activations)

    def _hook_activations(self, module, inputs, outputs):
        self.activations = outputs

    def grads(self, outputs, inputs, retain_graph=True, create_graph=True):
        grads = torch.autograd.grad(outputs=outputs,
                                    inputs=inputs,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)[0]
        self.model.zero_grad()
        return grads


class GradIntegral:
    def __init__(self, model, modules):

        self.modules = modules
        self.hooks = []

    def add_noise(self):
        for module in self.modules:
            hook = module.register_forward_hook(_modify_feature_map)
            self.hooks.append(hook)

    def remove_noise(self):
        for hook in self.hooks:
            hook.remove()
            self.hooks.clear()


# keep forward after modify
def _modify_feature_map(module, inputs, outputs):
    noise = torch.randn(outputs.shape).to(outputs.device)
    # noise = torch.normal(mean=0, std=3, size=outputs.shape).to(outputs.device)

    p = np.array([0.3, 0.7])
    random = torch.tensor([np.random.choice([0, 1], p=p.ravel()) for i in range(outputs.shape[1])]).to(outputs.device)
    random = torch.unsqueeze(torch.unsqueeze(random, -1), -1)
    noise = noise * random

    outputs += noise


def _test():
    import models
    from configs import config

    model = models.load_model('efficientnetv2s')
    model_path = r'{}/{}/checkpoint.pth'.format(config.output_model, 'efficientnetv2s_03271844')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    model.eval()

    gi = GradIntegral(model=model, modules=[model.features[40].conv[7]])
    module = HookModule(model=model, module=model.features[40].conv[7])

    inputs = torch.ones((4, 3, 224, 224))
    labels = torch.tensor([1, 1, 1, 1])

    print('-' * 10)
    gi.add_noise()
    outputs = model(inputs)
    print(outputs)

    print('-' * 10)
    gi.remove_noise()
    outputs = model(inputs)
    print(outputs)

    print('-' * 10)
    gi.add_noise()
    outputs = model(inputs)
    print(outputs)
    nll_loss = torch.nn.NLLLoss()(outputs, labels)
    grads = module.grads(outputs=-nll_loss, inputs=module.activations)
    print(grads)


if __name__ == '__main__':
    _test()

    # 9, 224, 224 -> 27, 112, 112 -> 81, 56, 56
    # noise = torch.randn(size=(4, 81, 56, 56))
    # print(noise)
    # noise = torch.normal(mean=0, std=3, size=(4, 81, 56, 56))
    # print(noise)
