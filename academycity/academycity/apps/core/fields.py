from django.db import models
from django.core.exceptions import ObjectDoesNotExist


class OrderField(models.PositiveIntegerField):
    def __init__(self, for_fields=None, *args, **kwargs):
        self.for_fields = for_fields
        super(OrderField, self).__init__(*args, **kwargs)

    def pre_save(self, model_instance, add):
        # print('----1000----')
        # print(self.attname)
        # print(model_instance)
        # print(getattr(model_instance, self.attname))
        #
        # print('----1000----')

        if getattr(model_instance, self.attname) is None or getattr(model_instance, self.attname) == 1:
            # no current value
            # print('----10001----')
            try:
                qs = self.model.objects.all()
                if self.for_fields:
                    # filter by objects with the same field values
                    # for the fields in "for_fields"
                    query = {field: getattr(model_instance, field) for field in self.for_fields}

                    # print(query)

                    qs = qs.filter(**query).order_by("order")
                # get the order of the last item
                last_item = qs.latest(self.attname)
                # print(last_item)
                #
                # print(last_item.order)

                value = last_item.order + 1
            except ObjectDoesNotExist:
                value = 1
            setattr(model_instance, self.attname, value)
            return value
        else:
            return super(OrderField,
                         self).pre_save(model_instance, add)
