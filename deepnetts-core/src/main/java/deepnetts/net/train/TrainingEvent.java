/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.net.train;

/**
 * This class holds source and type of training event.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public final class TrainingEvent {

    private final BackpropagationTrainer source;
    private final Type type;

    public static enum Type {
        STARTED, STOPPED, EPOCH_FINISHED, MINI_BATCH, ITERATION_FINISHED;
    }

    public TrainingEvent(final BackpropagationTrainer source, final Type type) {
        this.source = source;
        this.type = type;
    }

    public BackpropagationTrainer getSource() {
        return source;
    }

    public Type getType() {
        return type;
    }

//    public static Type STARTED = Type.STARTED;
//    public static Type STOPPED = Type.STOPPED;
//    public static Type EPOCH_FINISHED = Type.EPOCH_FINISHED;
//    public static Type MINI_BATCH = Type.MINI_BATCH;
//    public static Type ITERATION_FINISHED = Type.ITERATION_FINISHED;

}