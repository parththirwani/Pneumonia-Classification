#Classification Report

# Generate a classification report
report = classification_report(all_labels, all_preds, target_names=class_names)

# Print the classification report
print(report)