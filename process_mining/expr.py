
"""
This module is used to implement the Expression class.
The expression class parses an expression from a DCR graph and is able to interpret that
"""
import numbers
from enum import Enum, auto


class Expression(object):
    """
    Boolean expression that represents a guard
    """
    compare_operators: [] = ['<', '>', '=', '!']

    def __init__(self, expression: str = str, expression_id='unit-test'):
        """
        The constructor for an expression
        :param expression: The expression string
        :param expression_id: the id of the expression in the xml, can be unit-test to indicate testing
        """
        # Initialization of the members
        self.expression_id = expression_id
        self.expression_str: str = expression
        self.expression_left: str = str()
        self.expression_right = str()
        self.expression_middle = str()
        self.CheckDifferentEvent: bool = False
        self.CompareWithEventData: bool = False
        self.ReferenceNode = None
        self.expression_reference_node: str = None
        self.expression_comparator: Comparators = None

        # If not unit-test split it up
        if expression_id != 'unit-test':
            self.split_singular_expression(expression)
            self.CheckDifferentEvent = self.check_if_data_in_different_event()
            self.try_convert_expression_right()
            self.convert_comparator()

    def try_convert_expression_right(self):
        """
        Data comes as string but may be float or int, thus comparison needs to be done
        :return:
        """
        if self.expression_right.strip().startswith('{') and self.expression_right.endswith('}'):
            self.CompareWithEventData = True
            self.expression_right = self.expression_right.replace('{', '').replace('}', '')
        else:
            self.expression_right = try_convert_value(self.expression_right)

    def split_singular_expression(self, expression):
        """
        Splite the expression in three evaluable parts
        :param expression:
        :return:
        """
        index = 0
        expression_tmp = expression.strip()
        for char in expression_tmp:
            if char in Expression.compare_operators:
                break
            self.expression_left = self.expression_left + char
            index += 1
        self.expression_left = self.expression_left.strip()
        expression_tmp = expression_tmp[index:]
        index = 0
        for char in expression_tmp:
            if char in Expression.compare_operators:
                self.expression_middle = self.expression_middle + char
                index += 1
            elif char is ' ':
                index += 1
            else:
                break
        self.expression_right = expression_tmp[index:].strip()

    def evaluate_expression(self, event, trace_data=None):
        """
        Checks the guard, and if the guard is active true is returned
        :param event: The current event of the Trace used to evaluate the expression
        :param trace_data: The trace_conformance_data_object
        :return:
        """
        right_expr = self.expression_right
        if self.expression_right in event.Attributes and self.CompareWithEventData:
            right_expr = event.Attributes[self.expression_right].get_value()
        # Check if the expression is related to a different event
        if self.CheckDifferentEvent and trace_data is not None:
            previously_executed_event = self.try_get_previous_execution(event, trace_data)
            if previously_executed_event is None:
                return False
            if self.expression_left in previously_executed_event.Attributes:
                expression_left_value = \
                    previously_executed_event.Attributes[self.expression_left].get_value()
                return self.do_comparison(expression_left_value, right_expr)
            else:
                return False
        if self.expression_left in event.Attributes:
            expression_left_value = event.Attributes[self.expression_left].get_value()
            return self.do_comparison(expression_left_value, right_expr)
        # If you only want to check for the existence and the Attribute is not even in the event_log
        elif self.expression_left not in event.Attributes and self.expression_right is '':
            if self.expression_comparator == Comparators.neq:
                return False
            elif self.expression_comparator == Comparators.eq:
                return True


    def convert_comparator(self):
        """
        Convert the string of the evaluator to an enum that makes distinction possible
        :return:
        """
        if self.expression_middle == '>':
            self.expression_comparator = Comparators.gt

        elif self.expression_middle == '<':
            self.expression_comparator = Comparators.lt

        elif self.expression_middle in ['>=', '=>']:
            self.expression_comparator = Comparators.gte

        elif self.expression_middle in ['<=', '=<']:
            self.expression_comparator = Comparators.lte

        elif self.expression_middle in ['=', '==']:
            self.expression_comparator = Comparators.eq

        elif self.expression_middle in ['!', '!=', '!!', '=!']:
            self.expression_comparator = Comparators.neq
        else:
            raise ValueError('The value {} could not be converted to a comparator'
                             .format(self.expression_middle))

    def check_if_data_in_different_event(self):
        """
        Check whether the variable is like "a.data>1"
        :return: if data of different event return true, else return false
        """
        if '.' in self.expression_left:
            self.expression_reference_node = self.expression_left.split('.')[0]
            self.expression_left = self.expression_left.split('.')[1]
            return True
        return False

    def split_multiple_expression(self, expression):
        """
        Splits a multiple expression to singular expressions
        For example "data>10 && data2<10" or
        :param expression: The expression to be split
        :return: Not Implemented Error
        TODO: Future Feature
        """
        raise NotImplementedError

    def do_comparison(self, left_value, right_expr):
        """
        Performs the comparison given within an expression
        :param left_value: The left value that was retrieved from an event-log
        :return: True if expression true, false if not
        """
        # First checking it they types of the values that are compared are equal
        if right_expr is "":
            if self.expression_comparator == Comparators.neq:
                return bool(left_value)
            elif self.expression_comparator == Comparators.eq:
                return not bool(left_value)
        if not isinstance(left_value, type(right_expr)) and not \
                isinstance(left_value, numbers.Real) == isinstance(right_expr, numbers.Real):

            # Fallback implementation, when this happens the data type could not be parsed from the event log itself
            left_value = try_convert_value(left_value)

            # If even after fallback data type is not equal, then if comparator says not eq True.
            if not isinstance(left_value, type(right_expr)) and not \
                    isinstance(left_value, numbers.Real) == isinstance(right_expr, numbers.Real):
                if self.expression_comparator == Comparators.neq:
                    return True
                else:
                    return False
        # Perform comparison of the data
        if self.expression_comparator == Comparators.gte:
            return left_value >= right_expr
        elif self.expression_comparator == Comparators.gt:
            return left_value > right_expr
        elif self.expression_comparator == Comparators.eq:
            return left_value == right_expr
        elif self.expression_comparator == Comparators.lt:
            return left_value < right_expr
        elif self.expression_comparator == Comparators.lte:
            return left_value <= right_expr
        elif self.expression_comparator == Comparators.neq:
            return left_value != right_expr
        else:
            raise TypeError("Not a valid comparator")

    def set_reference_node(self, node):
        """
        TODO Check if needed
        Sets the node of the DCRGraph as the reference node.
        Mainly used to check if the mentioned event is in the graph
        :param node: The node that is set
        :return:
        """
        self.ReferenceNode = node

    def try_get_previous_execution(self, event, trace_data):
        """
        Get the last execution of the referenced event
        :param event
        :param trace_data:
        :return:
        """
        # get the index of
        index = trace_data.Trace.Events.index(event)
        while index > 0:
            var = trace_data.Trace.Events[index - 1]
            if str(var.EventName) == self.expression_reference_node:
                return var
            index -= 1
        return None


class Comparators(Enum):
    """
    Different types to be used to differentiate comparisons
    """
    eq = auto()  # Checks Equality
    neq = auto()  # Checks Inequality
    lt = auto()  # lower than
    gt = auto()  # greater than
    gte = auto()  # greater than or equal
    lte = auto()  # lower than or equal
    exists=auto()
    # ... Add up more if required


def try_convert_value(value:str):
    """
    The real conversion method. always with try except to make it error prone, fallback is string
    :param value: the value which should be converted in the data type
    :return:
    """
    if type(value) is not str:
        raise TypeError("value to be converted was not a string")
    if value.lower() in ['true', 't', 'y']:
        return True
    elif value.lower() in ['false', 'f', 'n']:
        return False

    # Try conversion to int
    try:
        # Justification: Flexible type
        # noinspection PyTypeChecker
        return int(value)
    except ValueError:
        # Error is passed to make next conversion
        # print(type(e))
        pass

    # Try conversion to float
    try:
        if ',' in value:
            return float(value.replace(',', '.'))
        return float(value)
    except ValueError:
        # Error is passed to make next conversion
        # print(type(e))
        pass
    return value

    # TODO Consider Date, Time and Datetime
    # # Try parse a date
    # try:
    #     datetime_object = datetime.strptime(value, '%Y/%m/%d')
    #     return datetime_object.date()
    # except:
    #     pass
    #
    # # Try parse time /w milliseconds
    # try:
    #     time_obj = datetime.strptime(right_expr, '%H:%M:%S.%f')
    #     self.expression_right = time_obj.time()
    # except:
    #     pass
    #
    # try:
    #     time_obj = datetime.strptime(self.expression_right, '%H:%M:%S')
    #     self.expression_right = time_obj.time()
    # except:
    #     pass
    #
    # try:
    #     datetime_obj = datetime.strptime(self.expression_right, '%Y/%m/%d %H:%M:%S.%f')
    #     self.expression_right = datetime_obj
    # except:
    #     pass
