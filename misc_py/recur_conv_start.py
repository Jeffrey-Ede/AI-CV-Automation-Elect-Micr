def embedding(input, features):
        embed = strided_conv_block(input, features)
        embed = strided_conv_block(input, features)
        return embed

def reconstruction(input, features):
    recon = strided_conv_block(input, features)
    recon = strided_conv_block(input, features)
    return recon

def recur_frac_conv(input, embed_scope=None, recur_scope=None, recon_scope=None, turns=turns):
    """Fractal recursive convolutions"""

    if recon_scope:
        with tf.variable_scope(recon_scope, reuse=True) as scope0:
            embed = embedding(input, resolved5)
    else:
        default_embed_scope="embedding"
        with tf.variable_scope(default_embed_scope, reuse=False) as scope0:
            embed = embedding(input, resolved5)

    #Perform recursive convolutions
    recur_convs = []
    if scope:
        with tf.variable_scope(recur_scope, reuse=True) as scope1:
            recur_frac_conv = strided_conv_block(embed, resolved5)
    else:
        default_recur_scope="fract_recur_conv"
        with tf.variable_scope(default_recur_scope, reuse=False) as scope1:
            recur_frac_conv = strided_conv_block(embed, resolved5)
    recur_convs.append(recur_frac_conv)

    for _ in range(1, turns):
        with tf.variable_scope(scope1, reuse=True):
            recur_frac_conv = strided_conv_block(recur_frac_conv, resolved5)
        recur_convs.append(recur_frac_conv)

    output = 0.
    for conv in recur_convs:
        concat = tf.concat([input, concat], axis=3)
                
        if recon_scope:
            with tf.variable_scope(recon_scope, reuse=True) as scope2:
                output += reconstruction(concat, resolved5)
        else:
            default_recur_scope="reconstruction"
            with tf.variable_scope(default_recon_scope, reuse=False) as scope2:
                output += reconstruction(concat, resolved5)

    return output, scope0, scope1, scope2
