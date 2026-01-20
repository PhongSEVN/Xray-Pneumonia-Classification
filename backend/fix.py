import tensorflow as tf

old_model = tf.keras.models.load_model("pneumonia_densenet121.keras", compile=False)

if isinstance(old_model.inputs, (list, tuple)) and isinstance(old_model.inputs[0], (list, tuple)):
    real_input = old_model.inputs[0][0]
else:
    real_input = old_model.inputs[0]

fixed_model = tf.keras.Model(inputs=real_input, outputs=old_model.outputs)
fixed_model.save("pneumonia_densenet121_fixed.keras", include_optimizer=False)

print("Model đã được sửa và lưu thành pneumonia_densenet121_fixed.keras")
