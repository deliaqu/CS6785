import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

def generate_data(n_samples, means, cov):
    data = np.vstack([
        np.random.multivariate_normal(mean, cov, n_samples) for mean in means
    ])
    return data

def plot_data(data, title, mus=None, output_file=None):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
    if mus is not None:
        plt.scatter(mus[:, 0].detach().numpy(), mus[:, 1].detach().numpy(), c='red', marker='x', s=100, label='Estimated Means')
        plt.legend()
    plt.title(title)

    if output_file is not None:
        plt.savefig(output_file)
    
    plt.show()
    plt.close()

def log_likelihood(data, mus):
    distributions = [MultivariateNormal(mu, torch.eye(2)) for mu in mus]
    log_probs = torch.stack([dist.log_prob(data) for dist in distributions], dim=1)
    log_likelihoods = torch.logsumexp(log_probs, dim=1) - np.log(3)
    return log_likelihoods.sum()

def gradient_ascent(data_tensor, mus, n_iterations, learning_rate):
    log_likelihoods = []
    for _ in range(n_iterations):
        ll = log_likelihood(data_tensor, mus)
        ll.backward()

        with torch.no_grad():
            mus.data += learning_rate * mus.grad
        mus.grad.zero_()

        log_likelihoods.append(ll.item()/data_tensor.size(0))
    return log_likelihoods

def sgd(data_tensor, sgd_mus, n_iterations, learning_rate, batch_size):
    sgd_log_likelihoods = []
    for _ in range(n_iterations):
        sample = data_tensor[np.random.choice(data_tensor.size(0), batch_size, replace=False)]
        
        ll = log_likelihood(sample, sgd_mus)
        ll.backward()
        
        with torch.no_grad():
            sgd_mus.data += learning_rate * sgd_mus.grad  # SGD update step
        sgd_mus.grad.zero_()
        
        sgd_log_likelihoods.append(ll.item()/batch_size)
    return sgd_log_likelihoods

def e_step(data, mus, pi):
    distributions = [MultivariateNormal(mu, torch.eye(2)) for mu in mus]
    probs = torch.stack([pi[k] * torch.exp(dist.log_prob(data)) for k, dist in enumerate(distributions)], dim=1)
    responsibilities = probs / probs.sum(dim=1, keepdim=True)
    return responsibilities

def m_step(data, pi, responsibilities):
    mus_new = (responsibilities.T @ data) / (data.shape[0] * pi.unsqueeze(1))
    return mus_new

def em(data, em_mus, em_pi, n_iterations):
    em_log_likelihoods = []
    for _ in range(n_iterations):
        responsibilities = e_step(data, em_mus, em_pi)
        em_mus.data = m_step(data, em_pi, responsibilities)
        
        ll = log_likelihood(data, em_mus)
        em_log_likelihoods.append(ll.item()/data.size(0))
    return em_log_likelihoods

def plot_log_likelihood(log_likelihoods, output_file=None):
    plt.plot(range(len(log_likelihoods)), log_likelihoods, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood Progression")

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()
    plt.close()

def main():
    np.random.seed(42)
    n_samples = 1000
    means = [
        np.array([-2, 0]),
        np.array([1, 3]),
        np.array([2, -2])
    ]
    cov = np.eye(2)

    data = generate_data(n_samples, means, cov)
    plot_data(data, "Scatter plot of sampled data from GMM", output_file="data.png")

    data_tensor = torch.tensor(data, dtype=torch.float32)
    torch.manual_seed(42)
    mus = torch.nn.Parameter(torch.randn(3, 2))

    ga_n_iterations = 10
    ga_learning_rate = 1e-3

    log_likelihoods = gradient_ascent(data_tensor, mus, ga_n_iterations, ga_learning_rate)
    plot_log_likelihood(log_likelihoods, output_file="log_likelihood_ga.png")
    plot_data(data, "Final estimated means using gradient ascent", mus, output_file="final_estimated_means_ga.png")
    print("Final estimated means using gradient ascent:", mus.detach().numpy())

    sgd_n_iterations = 5000
    sgd_learning_rate = 1e-3
    batch_size = 6
    sgd_mus = torch.nn.Parameter(torch.randn(3, 2))

    sgd_log_likelihoods = sgd(data_tensor, sgd_mus, sgd_n_iterations, sgd_learning_rate, batch_size)
    plot_log_likelihood(sgd_log_likelihoods, output_file="log_likelihood_sgd.png")
    plot_data(data, "Final estimated means using sgd", sgd_mus, output_file="final_estimated_means_sgd.png")
    print("Final estimated means using sgd:", sgd_mus.detach().numpy())


    em_n_iterations = 10
    em_pi = torch.nn.Parameter(torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32))
    em_mus = torch.nn.Parameter(torch.randn(3, 2))
    em_log_likelihoods = em(data_tensor, em_mus, em_pi, em_n_iterations)
    plot_log_likelihood(em_log_likelihoods, output_file="log_likelihood_em.png")
    plot_data(data, "Final estimated means using EM", em_mus, output_file="final_estimated_means_em.png")
    print("Final estimated means using EM:", em_mus.detach().numpy())

if __name__ == "__main__":
    main()