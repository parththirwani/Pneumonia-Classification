# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None, resize=None):
        self.data = []  # List to store data samples
        self.targets = []  # List to store corresponding labels
        self.class_to_int = {}  # Dictionary to map class labels to integers
        self.int_to_class = {}  # Dictionary to map integers to class labels
        class_index = 0
        
        # Iterate over each class in the directory
        for class_label in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_label)
            
            # Add class_label to class_to_int dictionary
            self.class_to_int[class_label] = class_index
            self.int_to_class[class_index] = class_label
            class_index += 1
            
            # Iterate over each image in the class directory
            for image_file in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_file)
                
                # Add the image path and corresponding label to the lists
                self.data.append(image_path)
                self.targets.append(self.class_to_int[class_label])
        
        self.transform = transform
        self.resize = resize

    def __getitem__(self, index):
        # Retrieve the data sample and its label based on the index
        data_sample = self.data[index]
        target = self.targets[index]

        # Load the image using PIL and convert to RGB
        image = Image.open(data_sample).convert('RGB')

        # Resize the image if specified
        if self.resize:
            image = self.resize(image)

        # Apply the data transformation if specified
        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)


# Define the Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Set the path to your custom dataset directory
data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/'

# Define the data transformation for training and validation
transform_train = Compose([
    RandomRotation(60),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_val = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define the resize transformation
resize = Resize((128, 128))

# Create an instance of your custom dataset with the resize transformation
train_dataset = CustomDataset(data_dir, transform=transform_train, resize=resize)

# Calculate the class weights using the 'balanced' option
unique_targets = list(set(train_dataset.targets))
class_weights = compute_class_weight('balanced', classes=unique_targets, y=train_dataset.targets)
weights = torch.Tensor(class_weights)

# Create a WeightedRandomSampler to oversample the minority class
sampler = WeightedRandomSampler(weights, len(weights))

# Set batch size
batch_size = 64

# Create data loaders for train and validation sets with the weighted sampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

# Split the training dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create data loaders for train and validation sets without sampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the data transformation for the test dataset (similar to transform_val)
transform_test = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create an instance of the test dataset using the CustomDataset class with transform_test
test_data_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'
test_dataset = CustomDataset(test_data_dir, transform=transform_test, resize=resize)

# Create a DataLoader for the test dataset
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Load the pre-trained MobileNet model
mobilenet = models.mobilenet_v2(pretrained=True)

# Modify the classifier to match the number of classes in your dataset
num_classes = len(unique_targets)
mobilenet.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

# Define the loss function and optimizer
criterion1 = FocalLoss(alpha=1, gamma=2)  # Use the Focal Loss
criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adam(mobilenet.parameters(), lr=0.001)

   
optimizer = optim.Adam(mobilenet.parameters(), lr=0.001)

# Train the model on your oversampled training data
num_epochs = 10  # You can adjust the number of epochs
for epoch in range(num_epochs):
    mobilenet.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = mobilenet(images)
        loss1 = criterion1(outputs, labels)
        loss2 = criterion2(outputs, labels)
        loss = 0.5 * loss1 + 0.5 * loss2 
        loss.backward()
        optimizer.step()

    # Validate the model on the validation dataset
    mobilenet.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = mobilenet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.2f}%')
    
# Evaluate the model on the test dataset
mobilenet.eval()
total = 0
correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = mobilenet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')