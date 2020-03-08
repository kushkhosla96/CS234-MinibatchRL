import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from CifarDataEvaluator import CifarDataEvaluator

DV_MODEL_PATH = 'cifar_agent_lbs512_sbs128_ebs64_ne4_ii120_maw15_evaluator.pt'

def load_trainloader():
	transform = transforms.Compose([transforms.ToTensor(), \
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, \
		download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, \
		shuffle=True, num_workers=2)
	return trainloader

def plot_hist(model, trainloader):
	all_weights = []
	for images, labels in trainloader:
		weights = model(images).detach().numpy()
		all_weights.extend(list(weights))
	plt.hist(all_weights, bins=20, range=(0, 1))
	plt.title("Weights from Data Evaluator Network")
	plt.xlabel("Weight (Probability of Choosing)")
	plt.show()
	plt.close()

def main():
	trainloader = load_trainloader()
	model = CifarDataEvaluator()
	model.load_state_dict(torch.load(DV_MODEL_PATH, map_location=torch.device('cpu')))
	plot_hist(model, trainloader)

if __name__ == "__main__":
	main()