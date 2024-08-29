from Generate_Function_Data import generate_function_data

def main():
    base_data_path = "/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Data/Generated_data_dim10_row10000"
    dim = 10
    num_samples = 10000
    generate_function_data(dim, num_samples, base_data_path)

    base_data_path = "/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Data/Generated_data_dim30_row25000"
    dim = 30
    num_samples = 25000
    generate_function_data(dim, num_samples, base_data_path)

    base_data_path = "/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Data/Generated_data_dim50_row50000"
    dim = 50
    num_samples = 50000
    generate_function_data(dim, num_samples, base_data_path)

main()