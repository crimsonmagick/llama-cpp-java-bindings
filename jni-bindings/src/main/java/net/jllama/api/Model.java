package net.jllama.api;

import net.jllama.core.LlamaModel;

public class Model {

  public Model(final LlamaModel llamaModel) {
    this.llamaModel = llamaModel;
  }

  private boolean closed;
  private LlamaModel llamaModel;


  public boolean isClosed() {
    return closed;
  }

  public LlamaModel getLlamaModel() {
    return llamaModel;
  }

}
